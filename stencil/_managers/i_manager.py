from __future__ import annotations
import logging
from typing import TYPE_CHECKING
import abc

if TYPE_CHECKING:
    from stencil import Service


class IManager(abc.ABC):
    def __init__(self, service: "Service"):
        self.service = service
        self._logger = logging.getLogger(self.__class__.__qualname__)
        self._log_area: str = (
            f"{service.__class__.__name__} [{self.__class__.__name__}]"
        )
