#

"""







"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from ..models import (
    catalogue_filter,
    catalogue_lookup,
    catalogue_providers_list,
    catalogue_providers_supported,
    catalogue_run_get,
    catalogue_run_list,
    catalogue_run_live,
)
from ..structs import LiveResult, ModelInfo
from ..types import Capability, Provider

if TYPE_CHECKING:
    from . import Client


class Models:
    """


"""

    def __init__(self, client: "Client", cap_filter: Capability | None = None) -> None:
        self.client = client
        self.cap_filter = cap_filter

    def with_capability(self, c: Capability) -> "Models":
        """
"""
        out = copy.copy(self)
        out.cap_filter = c
        return out

    def provider(self, p: Provider) -> "ScopedModels":
        """
"""
        return ScopedModels(self.client, p, self.cap_filter)

    def list(self) -> list[ModelInfo]:
        """
"""
        return catalogue_filter(self.cap_filter)

    def get(self, id: str) -> ModelInfo | None:
        """"""
        return catalogue_lookup(id)

    async def live(self) -> LiveResult:
        """
"""
        return await catalogue_run_live(self)


class ScopedModels:
    """

"""

    def __init__(
        self,
        client: "Client",
        target: Provider,
        cap_filter: Capability | None = None,
        raw_flag: bool = False,
    ) -> None:
        self.client = client
        self.target = target
        self.cap_filter = cap_filter
        self.raw_flag = raw_flag

    def raw(self) -> "ScopedModels":
        out = copy.copy(self)
        out.raw_flag = True
        return out

    async def list(self) -> list[ModelInfo]:
        return await catalogue_run_list(self)

    async def get(self, id: str) -> ModelInfo:
        return await catalogue_run_get(self, id)


class Providers:
    """

"""

    def __init__(self, client: "Client") -> None:
        self.client = client

    def list(self) -> list[Provider]:
        return catalogue_providers_list(self.client)

    def supported(self) -> list[Provider]:
        return catalogue_providers_supported()
