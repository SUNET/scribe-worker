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
import sys
import torch

import whisper_timestamped as whisper

from daemonize import Daemonize
from huggingface_hub import snapshot_download
from random import randint
from time import sleep
from typing import Optional
from utils.args import parse_arguments
from utils.log import get_fileno, get_logger
from utils.settings import get_settings


mp.set_start_method("spawn", force=True)
settings = get_settings()
logger = get_logger()
(
    foreground,
    pidfile,
    zap,
    _,
    _,
    _,
    no_healthcheck,
    download,
    drain,
    drainfile,
) = parse_arguments()

if not zap and not download and not drain:
    from utils.job import TranscriptionJob


def _gpu_payload(gpu) -> dict:
    return {
        "index": gpu.index,
        "name": gpu.name,
        "memory_used": gpu.memory_used,
        "memory_total": gpu.memory_total,
        "utilization": gpu.utilization,
        "temperature": gpu.temperature,
        "power_draw": gpu.power_draw,
    }


def healthcheck() -> None:
    hostname = os.uname()[1]

    while True:
        # Gather load average, memory usage and GPU usage
        load_avg = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent

        try:
            gpus = list(gpustat.GPUStatCollection.new_query())
        except Exception:
            gpus = []

        if len(gpus) > 1:
            # Multiple GPUs: report each one under its own worker_id so
            # per-GPU load can be tracked separately.
            health_data_list = [
                {
                    "worker_id": f"{hostname}-{gpu.index}",
                    "load_avg": load_avg,
                    "memory_usage": memory_usage,
                    "gpu_usage": [_gpu_payload(gpu)],
                }
                for gpu in gpus
            ]
        else:
            health_data_list = [
                {
                    "worker_id": hostname,
                    "load_avg": load_avg,
                    "memory_usage": memory_usage,
                    "gpu_usage": [_gpu_payload(gpu) for gpu in gpus],
                }
            ]

        for health_data in health_data_list:
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


def mainloop(worker_id: int, gpu_id: Optional[int] = None) -> None:
    """
    Main function to fetch jobs and process them.
    """

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(f"[{worker_id}] Pinned to GPU {gpu_id}")

    logger.info(f"[{worker_id}] Starting worker process...")

    api_url = f"{settings.API_BACKEND_URL}/api/{settings.API_VERSION}/job"

    while True:
        sleep_time = randint(5, 60)

        logger.debug(
            f"[{worker_id}] Sleeping for {sleep_time} seconds before fetching a new job..."
        )

        sleep(sleep_time)

        if os.path.exists(drainfile):
            logger.info(
                f"[{worker_id}] Drain file {drainfile} present, skipping new job."
            )
            continue

        with TranscriptionJob(
            logger,
            api_url,
            hf_token=settings.HF_TOKEN,
        ) as job:
            job.start()


def main() -> None:
    logger.info("Starting transcription service...")

    num_gpus = torch.cuda.device_count()
    logger.info(f"Detected {num_gpus} GPU(s)")

    if no_healthcheck:
        processes = []
    else:
        processes = [mp.Process(target=healthcheck)]

    processes += [
        mp.Process(
            target=mainloop,
            args=(i, i % num_gpus if num_gpus > 0 else None),
        )
        for i in range(settings.WORKERS)
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()


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


def drain_create() -> None:
    """Create drain file. Workers stop fetching new jobs while it exists."""
    try:
        with open(drainfile, "w") as f:
            f.write(str(os.getpid()))
        print(f"Drain file created at {drainfile}.")
        print("Workers will finish ongoing jobs and stop fetching new ones.")
        print(f"Remove the file to resume: rm {drainfile}")
    except OSError as e:
        print(f"Failed to create drain file {drainfile}: {e}")
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
    elif drain:
        drain_create()
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
            verbose=False,
            keep_fds=[get_fileno()],
            auto_close_fds=False,
            chdir=os.getcwd(),
        )
        daemon.start()
