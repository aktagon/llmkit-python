"""Public provider namespace (ADR-038).

Exposes the narrow public per-provider catalogue keyless (no client needed —
the headline use case is "which env var holds the key?", asked before a client
exists): ``providers.info(name)`` / ``providers.list()``. The internal
wire/transform spec stays private under ``.generated`` and is not re-exported.
"""

from .generated.provider_info import ProviderInfo, info, list

__all__ = ["ProviderInfo", "info", "list"]
