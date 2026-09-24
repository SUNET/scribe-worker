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
A model loaded into this process and kept there.

Unlike transcription, which runs each job in a throwaway subprocess so the
OS reclaims every byte afterwards, a language model stays resident: loading
tens of gigabytes of weights takes far longer than answering with them, and
a reader waiting for a summary would spend most of that wait on a load that
the previous request had already done.
"""

import os
import time

from typing import Callable, Iterator, Optional

from utils.llm.base import Provider, ProviderError, Usage
from utils.log import get_logger
from utils.settings import get_settings

logger = get_logger()
settings = get_settings()


class TransformersProvider(Provider):
    """
    A Hugging Face causal language model, resident on the GPU.

    Parameters:
        alias (str): Registry name.
        config (dict): Registry entry. model_id is the repository.
    """

    def __init__(self, alias: str, config: dict) -> None:
        super().__init__(alias, config)
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """
        Load the tokenizer and the weights.

        Returns:
            None

        Raises:
            ProviderError: When the model cannot be loaded.
        """

        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        if settings.HF_TOKEN:
            os.environ["HF_TOKEN"] = settings.HF_TOKEN

        logger.info(f"Loading inference model {self.alias} ({self.model_id})...")
        started = time.monotonic()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, dtype="auto", device_map="auto"
                )
            except TypeError:
                # transformers 4.x spells it torch_dtype; 5.x renamed it.
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, torch_dtype="auto", device_map="auto"
                )
        except Exception as e:
            # The reason travels with it. A load failure is about weights and
            # credentials, never about anything a reader wrote, and the first
            # line of it is usually the whole answer -- "You are trying to
            # access a gated repo", say. Swallowing it left the log saying
            # only that something had not worked.
            reason = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__

            raise ProviderError(
                f"Could not load model {self.alias} ({self.model_id}): {reason}"
            ) from e

        logger.info(
            f"Model {self.alias} loaded in {time.monotonic() - started:.1f}s."
        )

    def _render(self, system: str, prompt: str) -> str:
        """
        Apply the model's chat template to a system and a user prompt.

        Some templates -- Gemma's among them -- have no system role at all
        and raise rather than ignore one. The instructions still have to
        reach the model, so they are folded into the top of the user turn.

        Parameters:
            system (str): System prompt.
            prompt (str): User prompt.

        Returns:
            str: The rendered prompt.
        """

        try:
            return self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{system}\n\n{prompt}"}],
                add_generation_prompt=True,
                tokenize=False,
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
        Generate an answer, yielding it as the model produces it.

        Parameters:
            system (str): System prompt.
            prompt (str): User prompt, carrying the transcript.
            max_output_tokens (int): Generation cap.
            usage (Usage): Filled in as generation proceeds.
            should_stop (Optional[Callable]): Consulted between pieces.

        Yields:
            str: The next piece of the answer.
        """

        from threading import Thread
        from transformers import StoppingCriteria, StoppingCriteriaList
        from transformers import TextIteratorStreamer

        self.load()

        class _Cancelled(StoppingCriteria):
            """Stops generation when the reader has gone away."""

            def __call__(self, input_ids, scores, **kwargs) -> bool:
                return bool(should_stop and should_stop())

        rendered = self._render(system, prompt)
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)

        usage.input_tokens = int(inputs["input_ids"].shape[-1])

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation = {
            **inputs,
            "max_new_tokens": max_output_tokens,
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([_Cancelled()]),
            "do_sample": self.config.get("temperature", 0) > 0,
            "temperature": self.config.get("temperature", 0.3),
            "top_p": self.config.get("top_p", 0.95),
        }

        started = time.monotonic()
        thread = Thread(target=self.model.generate, kwargs=generation)
        thread.start()

        generated = 0

        try:
            for piece in streamer:
                if not piece:
                    continue

                generated += 1
                yield piece

                if should_stop and should_stop():
                    break
        finally:
            # The stopping criterion above is what actually ends generation;
            # joining keeps the next request from starting while the GPU is
            # still busy with this one.
            thread.join(timeout=60)
            usage.gpu_seconds = time.monotonic() - started
            usage.output_tokens = generated
