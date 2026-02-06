import asyncio
import json
import ssl

import ollama
import websockets

from utils.log import get_logger
from utils.settings import get_settings

log = get_logger()
settings = get_settings()


class InferenceClient:
    """Client for connecting to backend inference WebSocket and handling Ollama requests."""

    def __init__(self):
        self.ws = None
        self.ollama_client = ollama.Client(host=settings.OLLAMA_URL)

    async def connect(self):
        """Connect to the backend inference WebSocket."""
        log.info(f"Connecting to Ollama at {settings.OLLAMA_URL}")
        ws_url = settings.API_BACKEND_URL.replace("https://", "wss://").replace(
            "http://", "ws://"
        )

        log.info(
            f"Backend WebSocket URL: {ws_url}/api/{settings.API_VERSION}/job/inference"
        )
        ws_url = f"{ws_url}/api/{settings.API_VERSION}/job/inference"

        # For wss://, use SSL context with client certs if configured, otherwise use default SSL
        ssl_context = None
        if ws_url.startswith("wss://"):
            if settings.SSL_CERTFILE and settings.SSL_KEYFILE:
                ssl_context = ssl.create_default_context()
                ssl_context.load_cert_chain(settings.SSL_CERTFILE, settings.SSL_KEYFILE)
            else:
                ssl_context = True  # Use default SSL context without client certificates

        log.info(f"Connecting to inference websocket: {ws_url}")
        self.ws = await websockets.connect(
            ws_url,
            ssl=ssl_context,
            additional_headers={
                "x-client-dn": "CN=TranscriberWorker,O=SUNET,ST=Stockholm,C=SE"
            },
        )
        log.info("Connected to inference websocket")

    async def handle_request(self, data: dict) -> dict:
        """Handle an inference request by calling Ollama."""
        log.debug(f"Handling inference request: {data}")
        request_id = data.get("request_id")
        prompt = data.get("prompt", "")
        model = data.get("model", "gemma3")

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.ollama_client.chat(model=model, messages=messages)

            return {
                "request_id": request_id,
                "result": response.message.content,
                "model": model,
            }
        except Exception as e:
            log.error(f"Ollama error: {e}")
            return {"request_id": request_id, "error": str(e)}

    async def run(self):
        """Main loop to receive and process inference requests."""
        while True:
            try:
                await self.connect()
                async for message in self.ws:
                    data = json.loads(message)
                    log.debug(f"Received inference request: {data}")
                    response = await self.handle_request(data)
                    await self.ws.send(json.dumps(response))
                    log.debug(f"Sent inference response: {response}")
            except websockets.ConnectionClosed:
                log.warning("Inference websocket connection closed, reconnecting...")
                await asyncio.sleep(5)
            except Exception as e:
                log.error(f"Inference client error: {e}")
                await asyncio.sleep(5)


def start_inference_client():
    """Start the inference client in an asyncio event loop."""
    client = InferenceClient()
    asyncio.run(client.run())
