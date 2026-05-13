# pyright: reportUnsupportedDunderAll=false
"""Responsibility-oriented modules for the upmix CLI."""

from . import (
    cli,
    de_limiter,
    filters,
    mix_tuning,
    models,
    outputs,
    placement,
    recovery,
    reporting,
    stems,
    temporal,
    utils,
    validation,
)

_modules = (
    cli,
    de_limiter,
    filters,
    mix_tuning,
    models,
    outputs,
    placement,
    recovery,
    reporting,
    stems,
    temporal,
    utils,
    validation,
)

__all__: list[str] = []

for _module in _modules:
    for _name in getattr(_module, "__all__", ()):
        globals()[_name] = getattr(_module, _name)
        __all__.append(_name)

del _module, _modules, _name
