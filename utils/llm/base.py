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
What every language model backend looks like from the outside.

A provider is deliberately synchronous and streaming: it yields text as the
model produces it, and the worker's socket loop runs it in a thread. That
suits both cases -- a model resident in this process generates on the GPU
and cannot be awaited, and a local model server is read line by line.

Providers never log prompts or completions. A transcript is the reader's
material; it passes through this process and leaves no trace in it.
"""

import ipaddress
import socket

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterator, Optional
from urllib.parse import urlparse


@dataclass
class Usage:
    """
    What one generation cost, filled in by the provider as it goes.

    Parameters:
        input_tokens (int): Prompt tokens.
        output_tokens (int): Generated tokens.
        gpu_seconds (float): Time the model spent generating. For a resident
            model that is wall clock time on the GPU; for a model server it
            is how long the request took, which is the closest thing this
            side can honestly measure.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    gpu_seconds: float = 0.0


class ProviderError(Exception):
    """
    A generation could not be carried out.

    The message reaches the reader through the hub, so it says what went
    wrong and never quotes the request.
    """


class Provider(ABC):
    """
    One model, ready to generate.

    Parameters:
        alias (str): Registry name, as shown to readers and recorded in
            usage rows.
        config (dict): The registry entry for this alias.
    """

    def __init__(self, alias: str, config: dict) -> None:
        self.alias = alias
        self.config = config
        self.model_id = config.get("model_id", "")

    def load(self) -> None:
        """
        Prepare the model. Called once, before the first request.

        Returns:
            None
        """

    @abstractmethod
    def stream(
        self,
        system: str,
        prompt: str,
        max_output_tokens: int,
        usage: Usage,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Iterator[str]:
        """
        Generate an answer, yielding it in pieces.

        Parameters:
            system (str): System prompt.
            prompt (str): User prompt, carrying the transcript.
            max_output_tokens (int): Generation cap.
            usage (Usage): Filled in by the provider as it goes.
            should_stop (Optional[Callable]): Consulted between pieces. When
                it answers True the provider stops generating -- the reader
                cancelled, or closed the page.

        Yields:
            str: The next piece of the answer.
        """


def is_local_endpoint(url: str) -> bool:
    """
    Whether a model server address is on this machine or a private network.

    The service is self-hosted so that recordings and what is said in them
    stay here. A model endpoint that resolves to a public address would
    forward every transcript somewhere else, which is exactly the thing the
    architecture exists to prevent -- so this is checked before connecting,
    not documented and hoped for.

    Parameters:
        url (str): The base URL from the registry.

    Returns:
        bool: True when every address it resolves to is loopback, private,
            or link-local.
    """

    host = urlparse(url).hostname

    if not host:
        return False

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False

    addresses = {info[4][0] for info in infos}

    if not addresses:
        return False

    for address in addresses:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            return False

        if not (parsed.is_loopback or parsed.is_private or parsed.is_link_local):
            return False

    return True
