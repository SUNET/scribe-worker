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
The model registry: aliases from settings, resolved to providers.

Evaluating a second model is meant to cost one entry in INFERENCE_MODELS
and a restart. Nothing above this layer -- not the hub, not the frontend,
not the usage rows -- knows anything about a model beyond its alias.
"""

from typing import Optional

from utils.llm.base import Provider, ProviderError, Usage
from utils.log import get_logger
from utils.settings import get_settings

logger = get_logger()
settings = get_settings()

PROVIDERS = {
    "transformers": "utils.llm.transformers_provider:TransformersProvider",
    "openai": "utils.llm.openai_provider:OpenAIProvider",
}

# Providers are cached because a resident model must be loaded once, not
# once per request.
_instances: dict[str, Provider] = {}


def model_aliases() -> list[str]:
    """
    Every model this worker is configured to serve.

    Returns:
        list[str]: The aliases, sorted.
    """

    return sorted(settings.INFERENCE_MODELS.keys())


def get_provider(alias: str) -> Provider:
    """
    The provider for a model alias, loading it the first time.

    Parameters:
        alias (str): Registry name.

    Returns:
        Provider: The provider, ready to generate.

    Raises:
        ProviderError: When the alias is unknown or its entry is unusable.
    """

    if (provider := _instances.get(alias)) is not None:
        return provider

    if (config := settings.INFERENCE_MODELS.get(alias)) is None:
        raise ProviderError(f"Unknown model {alias}.")

    name = config.get("provider", "")

    if (path := PROVIDERS.get(name)) is None:
        raise ProviderError(f"Model {alias} names an unknown provider {name!r}.")

    module_name, class_name = path.split(":")
    module = __import__(module_name, fromlist=[class_name])

    provider = getattr(module, class_name)(alias, config)
    provider.load()

    _instances[alias] = provider

    return provider


def preload(alias: Optional[str] = None) -> None:
    """
    Load models up front so the first reader does not pay for it.

    Parameters:
        alias (Optional[str]): One alias, or None for every configured one.

    Returns:
        None
    """

    for name in [alias] if alias else model_aliases():
        try:
            get_provider(name)
        except ProviderError as e:
            # Full traceback at debug: the chained cause is where a gated
            # repository, a missing token or an out-of-memory says so.
            logger.error(f"Model {name} unavailable: {e}")
            logger.debug(f"Model {name} load failed", exc_info=True)


__all__ = [
    "Provider",
    "ProviderError",
    "Usage",
    "get_provider",
    "model_aliases",
    "preload",
]
