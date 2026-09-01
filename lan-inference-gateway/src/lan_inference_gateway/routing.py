"""Model-to-backend routing that keeps client requests runtime independent."""

from __future__ import annotations

from .config import BackendConfig, ConfigurationError, GatewaySettings


class UnknownModelError(LookupError):
    """Raised when no backend can serve a requested model."""


class BackendRegistry:
    def __init__(self, settings: GatewaySettings) -> None:
        settings.validate()
        self._backends = {backend.name: backend for backend in settings.backends}
        self._models: dict[str, BackendConfig] = {}

        for backend in settings.backends:
            for model in backend.models:
                if model in self._models:
                    previous = self._models[model].name
                    raise ConfigurationError(
                        f"Model {model!r} is configured for both {previous!r} and {backend.name!r}."
                    )
                self._models[model] = backend

        self._default = (
            self._backends[settings.default_backend]
            if settings.default_backend is not None
            else None
        )

    @property
    def backends(self) -> tuple[BackendConfig, ...]:
        return tuple(self._backends.values())

    def models(self) -> tuple[tuple[str, BackendConfig], ...]:
        return tuple(self._models.items())

    def resolve(self, model: str) -> BackendConfig:
        if model in self._models:
            return self._models[model]
        if self._default is not None:
            return self._default
        raise UnknownModelError(model)
