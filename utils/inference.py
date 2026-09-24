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

"""
The inference side of the worker.

Transcription jobs are fetched: the worker asks the backend for the next
one, every few seconds, forever. Inference is the other way round -- the
worker connects out to the hub and waits, and the hub pushes a request the
moment a reader asks for one. There is no queue to poll because there is no
queue on disk: an inference request exists only in the hub's memory and in
this process while it is being answered.

Nothing here writes anything down, and nothing here logs a prompt or an
answer. What passes through is somebody's recording.
"""

import asyncio
import json
import os
import ssl
import threading

from typing import Optional

import websockets

from utils.llm import ProviderError, Usage, get_provider, model_aliases
from utils.log import get_logger
from utils.settings import get_settings

logger = get_logger()
settings = get_settings()

# Generated text is sent on in small batches rather than a frame per token:
# a frame each would flood both the socket and the editor redrawing at the
# other end, for no visible gain -- this is faster than anyone reads.
FLUSH_SECONDS = 0.08
FLUSH_CHARS = 240


def worker_name() -> str:
    """
    How this worker announces itself to the hub.

    Returns:
        str: The configured name, or the hostname.
    """

    return settings.INFERENCE_WORKER_ID or os.uname()[1]


def ssl_context() -> Optional[ssl.SSLContext]:
    """
    The client certificate the hub authenticates this worker by.

    The same certificate the REST calls present. Without it the hub refuses
    the connection, which is deliberate: the socket carries transcripts.

    Returns:
        Optional[ssl.SSLContext]: The context, or None for a plain ws:// URL
            in development.
    """

    if not settings.INFERENCE_HUB_URL.startswith("wss://"):
        return None

    context = ssl.create_default_context()

    if settings.SSL_CERTFILE and settings.SSL_KEYFILE:
        context.load_cert_chain(settings.SSL_CERTFILE, settings.SSL_KEYFILE)

    return context


class InferenceWorker:
    """
    One connection to the hub, and the generations running under it.
    """

    def __init__(self) -> None:
        self.cancelled: set[str] = set()
        self.running: dict[str, asyncio.Task] = {}
        self.socket = None
        self._send_lock = asyncio.Lock()

    async def send(self, payload: dict) -> None:
        """
        Send one message to the hub.

        Parameters:
            payload (dict): The message.

        Returns:
            None
        """

        if self.socket is None:
            return

        try:
            async with self._send_lock:
                await self.socket.send(json.dumps(payload))
        except Exception:
            # The connection dropped. The reconnect loop deals with it; a
            # generation still running will find its answer going nowhere
            # and stop at the next check.
            pass

    async def run_forever(self) -> None:
        """
        Stay connected to the hub, reconnecting for as long as the process
        lives.

        Returns:
            None
        """

        if not settings.INFERENCE_HUB_URL:
            logger.error("INFERENCE_HUB_URL is not set, inference is disabled.")
            return

        models = model_aliases()

        if not models:
            logger.error("No models configured, inference is disabled.")
            return

        logger.info(
            f"Inference worker {worker_name()} serving {models}, "
            f"connecting to the hub."
        )

        while True:
            try:
                await self.session()
            except Exception as e:
                logger.error(f"Inference hub connection failed: {e}")

            await asyncio.sleep(settings.INFERENCE_RECONNECT_SECONDS)

    async def session(self) -> None:
        """
        One connection: hello, then work until the socket closes.

        Returns:
            None
        """

        headers = {"User-Agent": f"scribe-inference-worker/{worker_name()}"}

        try:
            connect = websockets.connect(
                settings.INFERENCE_HUB_URL,
                ssl=ssl_context(),
                additional_headers=headers,
                max_size=None,
                ping_interval=settings.INFERENCE_HEARTBEAT_SECONDS,
            )
        except TypeError:
            # websockets renamed extra_headers to additional_headers in 14.
            connect = websockets.connect(
                settings.INFERENCE_HUB_URL,
                ssl=ssl_context(),
                extra_headers=headers,
                max_size=None,
                ping_interval=settings.INFERENCE_HEARTBEAT_SECONDS,
            )

        async with connect as socket:
            self.socket = socket

            await self.send(
                {
                    "type": "hello",
                    "worker_id": worker_name(),
                    "models": model_aliases(),
                    "concurrency": settings.INFERENCE_CONCURRENCY,
                }
            )

            logger.info("Connected to the inference hub.")

            heartbeat = asyncio.create_task(self.heartbeat())

            try:
                async for raw in socket:
                    try:
                        message = json.loads(raw)
                    except ValueError:
                        continue

                    await self.handle(message)
            finally:
                heartbeat.cancel()
                self.socket = None

                for task in self.running.values():
                    task.cancel()

                self.running.clear()
                logger.info("Disconnected from the inference hub.")

    async def heartbeat(self) -> None:
        """
        Tell the hub we are still here.

        Returns:
            None
        """

        while True:
            await asyncio.sleep(settings.INFERENCE_HEARTBEAT_SECONDS)
            await self.send({"type": "heartbeat"})

    async def handle(self, message: dict) -> None:
        """
        Act on one message from the hub.

        Parameters:
            message (dict): The message.

        Returns:
            None
        """

        match message.get("type"):
            case "job":
                req_id = str(message.get("req_id", ""))

                if not req_id:
                    return

                self.running[req_id] = asyncio.create_task(self.generate(message))
            case "cancel":
                req_id = str(message.get("req_id", ""))
                self.cancelled.add(req_id)
                logger.info(f"Request {req_id} cancelled.")
            case "error":
                logger.warning(f"Hub reported: {message.get('message', '')}")
            case _:
                pass

    async def generate(self, job: dict) -> None:
        """
        Answer one request, streaming as the model produces it.

        The model runs in a thread -- generation is blocking, whether it is
        a GPU busy in this process or a model server being read line by
        line -- and pieces are handed back to the event loop as they come.

        Parameters:
            job (dict): The job message from the hub.

        Returns:
            None
        """

        req_id = str(job.get("req_id", ""))
        alias = str(job.get("model", ""))
        usage = Usage()

        logger.info(
            f"Request {req_id}: task={job.get('task')}, model={alias}, "
            f"{len(job.get('prompt', ''))} characters in"
        )

        try:
            provider = get_provider(alias)
        except ProviderError as e:
            await self.send({"type": "error", "req_id": req_id, "message": str(e)})
            self.finish(req_id)
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def should_stop() -> bool:
            return req_id in self.cancelled

        def produce() -> None:
            try:
                for piece in provider.stream(
                    system=str(job.get("system", "")),
                    prompt=str(job.get("prompt", "")),
                    max_output_tokens=int(job.get("max_output_tokens", 1024) or 1024),
                    usage=usage,
                    should_stop=should_stop,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, ("delta", piece))

                loop.call_soon_threadsafe(queue.put_nowait, ("done", ""))
            except ProviderError as error:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(error)))
            except Exception:
                # Never pass the exception text on: it can carry fragments
                # of whatever it was working on.
                logger.error(f"Request {req_id} crashed.", exc_info=True)
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("error", "The model failed to answer.")
                )

        threading.Thread(target=produce, daemon=True).start()

        pending: list[str] = []

        async def flush() -> None:
            if not pending:
                return

            text = "".join(pending)
            pending.clear()
            await self.send({"type": "delta", "req_id": req_id, "text": text})

        try:
            while True:
                try:
                    kind, payload = await asyncio.wait_for(
                        queue.get(), timeout=FLUSH_SECONDS
                    )
                except asyncio.TimeoutError:
                    await flush()
                    continue

                match kind:
                    case "delta":
                        pending.append(payload)

                        if sum(len(p) for p in pending) >= FLUSH_CHARS:
                            await flush()
                    case "done":
                        await flush()
                        await self.send(
                            {
                                "type": "done",
                                "req_id": req_id,
                                "input_tokens": usage.input_tokens,
                                "output_tokens": usage.output_tokens,
                                "gpu_seconds": round(usage.gpu_seconds, 3),
                            }
                        )
                        logger.info(
                            f"Request {req_id} done: {usage.output_tokens} tokens "
                            f"in {usage.gpu_seconds:.1f}s"
                        )
                        break
                    case "error":
                        await flush()
                        await self.send(
                            {"type": "error", "req_id": req_id, "message": payload}
                        )
                        break
        except asyncio.CancelledError:
            self.cancelled.add(req_id)
            raise
        finally:
            self.finish(req_id)

    def finish(self, req_id: str) -> None:
        """
        Forget a request once it is over.

        Parameters:
            req_id (str): The request identifier.

        Returns:
            None
        """

        self.cancelled.discard(req_id)
        self.running.pop(req_id, None)


def inference_loop() -> None:
    """
    Entry point for the inference worker process.

    Returns:
        None
    """

    # Checked before anything is loaded. Preloading first meant a worker
    # with no hub configured downloaded tens of gigabytes of weights and
    # only then reported that it had nowhere to send anything.
    if not settings.INFERENCE_HUB_URL:
        logger.error(
            "INFERENCE_HUB_URL is not set. The inference worker has nowhere "
            "to connect and will not start."
        )
        return

    if not model_aliases():
        logger.error("No models configured. The inference worker will not start.")
        return

    if settings.INFERENCE_PRELOAD:
        from utils.llm import preload

        preload()

    asyncio.run(InferenceWorker().run_forever())
