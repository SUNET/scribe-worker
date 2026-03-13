# Copyright (c) 2025-2026 Sunet.
# Contributor: Kristofer Hallin
#
# This file is part of Sunet Scribe.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gpustat
import multiprocessing as mp
import os
import psutil
import requests
import signal
import sys
import threading

import whisper_timestamped as whisper

from daemonize import Daemonize
from huggingface_hub import snapshot_download
from random import randint
from time import sleep
from utils.args import parse_arguments
from utils.log import get_fileno, get_logger
from utils.settings import get_settings


mp.set_start_method("spawn", force=True)
settings = get_settings()
logger = get_logger()
foreground, pidfile, zap, _, _, _, no_healthcheck, download = parse_arguments()
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"

if not zap and not download:
    from utils.job import TranscriptionJob


def _ignore_sigint():
    """Ignore SIGINT in child processes; the main process handles shutdown."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def healthcheck() -> None:
    _ignore_sigint()
    while True:
        # Gather load average, memory usage and GPU usage
        load_avg = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        gpu_usage = []

        try:
            gpu_stat = gpustat.GPUStatCollection.new_query()
        except Exception:
            gpu_stat = []

        for gpu in gpu_stat:
            gpu_usage.append(
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "memory_used": gpu.memory_used,
                    "memory_total": gpu.memory_total,
                    "utilization": gpu.utilization,
                    "temperature": gpu.temperature,
                    "power_draw": gpu.power_draw,
                }
            )

        health_data = {
            "worker_id": os.uname()[1],
            "load_avg": load_avg,
            "memory_usage": memory_usage,
            "gpu_usage": gpu_usage,
        }

        try:
            res = requests.post(
                f"{settings.API_BACKEND_URL}/api/{settings.API_VERSION}/healthcheck",
                json=health_data,
                cert=(settings.SSL_CERTFILE, settings.SSL_KEYFILE),
            )
            res.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Healthcheck failed: {e}")

        sleep(10)


def fail_job(uuid: str) -> None:
    """Mark a job as failed via the API."""
    api_url = f"{settings.API_BACKEND_URL}/api/{settings.API_VERSION}/job"
    try:
        response = requests.put(
            f"{api_url}/{uuid}",
            json={
                "status": "failed",
                "error": "Worker shutdown",
                "transcribed_seconds": None,
            },
            cert=(settings.SSL_CERTFILE, settings.SSL_KEYFILE),
        )
        response.raise_for_status()
        logger.info(f"Job {uuid}: marked as failed (shutdown)")
    except requests.RequestException as e:
        logger.error(f"Job {uuid}: failed to mark as failed: {e}")


def run_job(worker_id: int, active_jobs: dict) -> None:
    """
    Fetch and process a single transcription job.
    """
    _ignore_sigint()
    api_url = f"{settings.API_BACKEND_URL}/api/{settings.API_VERSION}/job"

    try:
        with TranscriptionJob(
            logger,
            api_url,
            hf_token=settings.HF_TOKEN,
            active_jobs=active_jobs,
        ) as job:
            job.start()
    except Exception:
        logger.exception(f"[{worker_id}] Worker crashed")


def main() -> None:
    logger.info("Starting transcription service...")

    manager = mp.Manager()
    active_jobs = manager.dict()

    hc = None
    if not no_healthcheck:
        hc = mp.Process(target=healthcheck)
        hc.start()

    shutdown_event = threading.Event()
    worker_id = 0
    workers: list[mp.Process] = []

    def shutdown(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    while not shutdown_event.is_set():
        # Reap finished workers
        workers = [w for w in workers if w.is_alive()]

        if len(workers) < settings.WORKERS:
            p = mp.Process(target=run_job, args=(worker_id, active_jobs))
            p.start()
            workers.append(p)
            worker_id += 1
        else:
            logger.info(f"All {settings.WORKERS} slots busy, waiting...")

        shutdown_event.wait(timeout=10)

    # Mark active jobs as failed
    for uuid in list(active_jobs.values()):
        fail_job(uuid)

    # Wait for workers to finish, then terminate stragglers
    for w in workers:
        w.join(timeout=10)
        if w.is_alive():
            logger.warning(f"Worker {w.pid} still alive, terminating...")
            w.terminate()
            w.join(timeout=5)

    if hc and hc.is_alive():
        hc.terminate()
        hc.join(timeout=5)

    manager.shutdown()
    logger.info("Shutdown complete.")


def daemon_kill() -> None:
    try:
        pid = int(open(pidfile, "r").read().strip())
        print(f"Zapping transcription service with PID {pid}...")
        os.kill(pid, 9)
        os.remove(pidfile)
    except FileNotFoundError:
        print("PID file not found, nothing to zap.")


def daemon_running() -> None:
    """
    Check if the daemon is running by checking the PID file.
    """
    if not os.path.exists(pidfile):
        return False

    try:
        with open(pidfile, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
    except FileNotFoundError:
        return
    except ProcessLookupError:
        os.remove(pidfile)
        return

    print(f"Daemon is already running with PID {pid}.")
    sys.exit(1)


def download_models() -> None:
    """Download all configured whisper models."""
    if settings.HF_TOKEN:
        os.environ["HF_TOKEN"] = settings.HF_TOKEN

    models = set()

    for lang_models in settings.WHISPER_MODELS_HF.values():
        for model_name in lang_models.values():
            models.add(model_name)

    print(sorted(models))

    for model_name in sorted(models):
        revision = None
        if "@" in model_name:
            model_name, revision = model_name.rsplit("@", 1)

        print(
            f"Downloading '{model_name}'"
            + (f" (revision: {revision})" if revision else "")
            + "..."
        )

        if "/" in model_name:
            kwargs = {"repo_id": model_name}
            if revision:
                kwargs["revision"] = revision
            kwargs["allow_patterns"] = [
                "*.json",
                "*.txt",
                "*.model",
                "*.safetensors",
                "*.bin",
            ]
            snapshot_download(**kwargs)
        else:
            whisper.load_model(model_name, device="cpu")

        print("  Done.")

    print("\nAll models downloaded.")


if __name__ == "__main__":
    if download:
        download_models()
    elif zap:
        daemon_kill()
    elif foreground:
        daemon_running()
        main()
    else:
        daemon_running()
        daemon = Daemonize(
            app="transcription_service",
            pid=pidfile,
            action=main,
            foreground=False,
            verbose=True,
            keep_fds=[get_fileno()],
            auto_close_fds=False,
            chdir=os.getcwd(),
        )
        daemon.start()
