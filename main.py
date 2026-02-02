import gpustat
import multiprocessing as mp
import os
import psutil
import requests
import sys

from daemonize import Daemonize
from random import randint
from time import sleep
from utils.args import parse_arguments
from utils.log import get_fileno, get_logger
from utils.settings import get_settings
from utils.whisper import diarization_init

settings = get_settings()
logger = get_logger()
foreground, pidfile, zap, _, _, _, no_healthcheck = parse_arguments()

if not zap:
    from utils.job import TranscriptionJob


def healthcheck() -> None:
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


def mainloop(worker_id: int) -> None:
    """
    Main function to fetch jobs and process them.
    """

    logger.info(f"Starting worker process {worker_id}...")

    api_url = f"{settings.API_BACKEND_URL}/api/{settings.API_VERSION}/job"
    drz = diarization_init(settings.HF_TOKEN)

    while True:
        sleep(randint(10, 60))

        with TranscriptionJob(
            logger,
            api_url,
            settings.FILE_STORAGE_DIR,
            hf_whisper=settings.HF_WHISPER,
            hf_token=settings.HF_TOKEN,
            diarization_object=drz,
        ) as job:
            job.start()


def main() -> None:
    logger.info("Starting transcription service...")

    if no_healthcheck:
        processes = []
    else:
        processes = [mp.Process(target=healthcheck)]

    processes += [
        mp.Process(target=mainloop, args=(i,)) for i in range(settings.WORKERS)
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


if __name__ == "__main__":
    if zap:
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
