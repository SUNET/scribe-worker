import gpustat
import multiprocessing as mp
import os
import psutil
import requests
import shutil
import signal
import tempfile

import torch

from huggingface_hub import snapshot_download
from random import randint
from time import sleep
from utils.args import parse_arguments
from utils.job import TranscriptionJob
from utils.log import get_logger
from utils.settings import get_settings


mp.set_start_method("spawn", force=True)
settings = get_settings()
logger = get_logger()
_, _, _, download = parse_arguments()


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
                timeout=10,
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
            timeout=10,
        )
        response.raise_for_status()
        logger.info(f"Job {uuid}: marked as failed (shutdown)")
    except requests.RequestException as e:
        logger.error(f"Job {uuid}: failed to mark as failed: {e}")


def run_job(worker_id: int, jobs_dir: str, work_dir: str, stop_event: mp.Event) -> None:
    """
    Fetch and process transcription jobs until stopped.
    """
    _ignore_sigint()

    api_url = f"{settings.API_BACKEND_URL}/api/{settings.API_VERSION}/job"

    # Stagger initial fetch to avoid all workers hitting the API at once
    stop_event.wait(timeout=randint(0, 10))

    while not stop_event.is_set():
        try:
            with TranscriptionJob(
                logger,
                api_url,
                hf_token=settings.HF_TOKEN,
                jobs_dir=jobs_dir,
                work_dir=work_dir,
            ) as job:
                job.start()
        except Exception:
            logger.exception(f"[{worker_id}] Worker crashed")

        stop_event.wait(timeout=randint(10, 60))


def main() -> None:
    logger.info("Starting transcription service...")

    work_dir = tempfile.mkdtemp(prefix="transcribe_worker_")
    os.chmod(work_dir, 0o700)
    jobs_dir = os.path.join(work_dir, "jobs")
    os.makedirs(jobs_dir, mode=0o700)

    hc = mp.Process(target=healthcheck)
    hc.start()

    stop_event = mp.Event()
    workers: list[mp.Process] = []

    for i in range(settings.WORKERS):
        p = mp.Process(target=run_job, args=(i, jobs_dir, work_dir, stop_event))
        p.start()
        workers.append(p)

    def shutdown(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        stop_event.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # Block until shutdown signal
    signal.pause()

    # Wait for workers to finish, then terminate stragglers
    for w in workers:
        w.join(timeout=15)
        if w.is_alive():
            logger.warning(f"Worker {w.pid} still alive, terminating...")
            w.terminate()
            w.join(timeout=5)

    # Mark active jobs as failed (after workers are done so job files are stable)
    for filename in os.listdir(jobs_dir):
        filepath = os.path.join(jobs_dir, filename)
        try:
            with open(filepath, "r") as f:
                uuid = f.read().strip()
            if uuid:
                fail_job(uuid)
            os.remove(filepath)
        except Exception:
            pass

    if hc.is_alive():
        hc.terminate()
        hc.join(timeout=5)

    shutil.rmtree(work_dir, ignore_errors=True)
    logger.info("Shutdown complete.")


def download_models() -> None:
    """Download all configured whisper models."""
    if settings.HF_TOKEN:
        os.environ["HF_TOKEN"] = settings.HF_TOKEN

    models = set()

    for lang_models in settings.WHISPER_MODELS_HF.values():
        for model_name in lang_models.values():
            models.add(model_name)

    for model_name in sorted(models):
        revision = None
        if "@" in model_name:
            model_name, revision = model_name.rsplit("@", 1)

        logger.info(
            f"Downloading '{model_name}'"
            + (f" (revision: {revision})" if revision else "")
            + "..."
        )

        repo_id = model_name if "/" in model_name else f"openai/whisper-{model_name}"
        kwargs = {"repo_id": repo_id}
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

        logger.info("  Done.")

    # Pre-download Silero VAD model
    logger.info("Downloading Silero VAD model...")
    torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    logger.info("  Done.")

    logger.info("\nAll models downloaded.")


if __name__ == "__main__":
    if download:
        download_models()
    else:
        main()
