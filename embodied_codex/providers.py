"""Explicit model provider and credential routing."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping


class ProviderConfigurationError(RuntimeError):
    pass


_OPENAI_ENDPOINT = "https://api.openai.com/v1"
_APEX_ENDPOINT = "https://api.apexin.ai/v1"


@dataclass(frozen=True)
class ProviderConfiguration:
    provider: str
    endpoint: str
    key_env: str
    api_key: str

    def redacted(self) -> dict[str, object]:
        return {"provider": self.provider, "endpoint": self.endpoint,
                "key_env": self.key_env, "configured": bool(self.api_key)}


def resolve_provider(*, provider: str | None = None, base_url: str | None = None,
                     environment: Mapping[str, str] | None = None) -> ProviderConfiguration:
    env = dict(os.environ if environment is None else environment)
    requested = str(provider or "").strip().casefold() or None
    if requested not in {None, "openai", "apex"}:
        raise ProviderConfigurationError(f"unsupported model provider: {provider}")
    if requested is None and base_url:
        raise ProviderConfigurationError(
            "a custom model endpoint requires an explicit --provider")
    configured = [name for name, key in (("openai", "OPENAI_API_KEY"),
        ("apex", "APEX_API_KEY")) if env.get(key)]
    if requested is None:
        if len(configured) > 1:
            raise ProviderConfigurationError(
                "multiple model credentials are configured; select --provider explicitly")
        if not configured:
            raise ProviderConfigurationError(
                "configure OPENAI_API_KEY or APEX_API_KEY, or pass --model package:Model")
        requested = configured[0]
    key_env = "OPENAI_API_KEY" if requested == "openai" else "APEX_API_KEY"
    key = str(env.get(key_env) or "")
    if not key:
        raise ProviderConfigurationError(f"{key_env} is not configured for provider {requested}")
    default_endpoint = _OPENAI_ENDPOINT if requested == "openai" else _APEX_ENDPOINT
    configured_endpoint = str(base_url or (
        env.get("APEX_BASE_URL") if requested == "apex" else default_endpoint) or "").rstrip("/")
    endpoint = configured_endpoint or default_endpoint
    if not endpoint.startswith("https://"):
        raise ProviderConfigurationError("model endpoint must use HTTPS")
    if endpoint != default_endpoint:
        raise ProviderConfigurationError(
            f"provider {requested} only permits its official endpoint {default_endpoint}")
    return ProviderConfiguration(requested, endpoint, key_env, key)


__all__ = ["ProviderConfiguration", "ProviderConfigurationError", "resolve_provider"]
