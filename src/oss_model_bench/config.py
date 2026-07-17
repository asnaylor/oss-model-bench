from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


class ConfigError(ValueError):
    """Raised when required benchmark configuration is missing or invalid."""


@dataclass(frozen=True)
class TargetConfig:
    base_url: str
    model: str
    tokenizer: str
    context_limit: int
    results_dir: Path
    api_key: str | None = None

    @classmethod
    def from_env(cls, *, require_key: bool = False) -> "TargetConfig":
        base_url = os.getenv("OMB_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        model = os.getenv("OMB_MODEL")
        tokenizer = os.getenv("OMB_TOKENIZER") or model
        api_key = os.getenv("OMB_API_KEY") or os.getenv("OPENAI_API_KEY")
        context_raw = os.getenv("OMB_CONTEXT_LIMIT", "131072")
        results_dir = Path(os.getenv("OMB_RESULTS_DIR", "results"))

        missing = [
            name
            for name, value in (("OMB_BASE_URL", base_url), ("OMB_MODEL", model), ("OMB_TOKENIZER", tokenizer))
            if not value
        ]
        if require_key and not api_key:
            missing.append("OMB_API_KEY")
        if missing:
            raise ConfigError(f"missing required configuration: {', '.join(missing)}")
        try:
            context_limit = int(context_raw)
        except ValueError as exc:
            raise ConfigError("OMB_CONTEXT_LIMIT must be an integer") from exc
        if context_limit < 4096:
            raise ConfigError("OMB_CONTEXT_LIMIT must be at least 4096")

        return cls(
            base_url=normalize_base_url(str(base_url)),
            model=str(model),
            tokenizer=str(tokenizer),
            context_limit=context_limit,
            results_dir=results_dir,
            api_key=api_key,
        )

    def public_dict(self) -> dict[str, object]:
        values = asdict(self)
        values.pop("api_key", None)
        values["results_dir"] = str(self.results_dir)
        return values

    def endpoint(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


def normalize_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError("OMB_BASE_URL must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise ConfigError("credentials must not be embedded in OMB_BASE_URL")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
