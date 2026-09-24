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
Any local model server that speaks the OpenAI chat API: vLLM, llama.cpp's
server, Ollama. One provider covers all three, which is what makes trying a
different model a line of configuration rather than a day of plumbing.

Local is enforced, not assumed -- see is_local_endpoint(). The reason the
service runs its own models at all is that transcripts must not leave; a
base_url pointing anywhere public would quietly undo that.
"""

import json
import time

from typing import Callable, Iterator, Optional

import requests

from utils.llm.base import Provider, ProviderError, Usage, is_local_endpoint
from utils.log import get_logger
from utils.settings import get_settings

logger = get_logger()
settings = get_settings()


class OpenAIProvider(Provider):
    """
    A model behind a local OpenAI-compatible endpoint.

    Parameters:
        alias (str): Registry name.
        config (dict): Registry entry. Needs base_url and model_id.
    """

    def load(self) -> None:
        """
        Check the endpoint before anything is sent to it.

        Returns:
            None

        Raises:
            ProviderError: When no base_url is configured, or when it is not
                local and remote endpoints have not been explicitly allowed.
        """

        base_url = self.config.get("base_url", "")

        if not base_url:
            raise ProviderError(f"Model {self.alias} has no base_url configured.")

        if not settings.INFERENCE_ALLOW_REMOTE and not is_local_endpoint(base_url):
            raise ProviderError(
                f"Model {self.alias} points at a non-local endpoint. "
                "Transcripts may not leave this network."
            )

    def stream(
        self,
        system: str,
        prompt: str,
        max_output_tokens: int,
        usage: Usage,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Iterator[str]:
        """
        Generate an answer, yielding it as the server streams it back.

        Parameters:
            system (str): System prompt.
            prompt (str): User prompt, carrying the transcript.
            max_output_tokens (int): Generation cap.
            usage (Usage): Filled in as generation proceeds.
            should_stop (Optional[Callable]): Consulted between pieces.

        Yields:
            str: The next piece of the answer.

        Raises:
            ProviderError: When the server refuses or cannot be reached.
        """

        self.load()

        base_url = self.config.get("base_url", "").rstrip("/")
        started = time.monotonic()
        generated = 0

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_output_tokens,
            "temperature": self.config.get("temperature", 0.3),
            "top_p": self.config.get("top_p", 0.95),
            "stream": True,
            # vLLM and llama.cpp both answer with a final usage block when
            # asked; servers that do not simply leave the counts at what we
            # counted ourselves.
            "stream_options": {"include_usage": True},
        }

        headers = {"Content-Type": "application/json"}

        if key := self.config.get("api_key"):
            headers["Authorization"] = f"Bearer {key}"

        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
                timeout=self.config.get("timeout", 600),
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise ProviderError(f"Model {self.alias} could not be reached.") from e

        try:
            # Bytes, decoded here rather than by requests. A server that
            # answers text/event-stream without naming a charset gets
            # ISO-8859-1 under HTTP's own rules, which is what requests
            # applies -- and every Swedish character in the answer came back
            # as mojibake ("Ã¥" for "å"). Both SSE and JSON are UTF-8, so it
            # is decoded as UTF-8, and a chunk that somehow is not does not
            # take the whole answer down with it.
            for raw in response.iter_lines():
                line = raw.decode("utf-8", errors="replace") if raw else ""

                if should_stop and should_stop():
                    break

                if not line or not line.startswith("data:"):
                    continue

                data = line[len("data:") :].strip()

                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue

                if counts := chunk.get("usage"):
                    usage.input_tokens = int(counts.get("prompt_tokens", 0) or 0)
                    usage.output_tokens = int(
                        counts.get("completion_tokens", 0) or usage.output_tokens
                    )

                for choice in chunk.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content")

                    if piece:
                        generated += 1
                        yield piece
        finally:
            response.close()
            usage.gpu_seconds = time.monotonic() - started

            if not usage.output_tokens:
                usage.output_tokens = generated
