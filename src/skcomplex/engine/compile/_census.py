"""Take the census of the laws of a system: the laws every component declares, with their owners and targets."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass, field

from sage.all__sagemath_symbolics import Expression

from ...core.errors import SkcomplexError
from ...core.system import System
from ._census_record import CensusRecord


@dataclass(frozen=True)
class Census:
    """Laws of every component of a system, and the errors of the components that fail to declare or collect.

    Args:
        system:
            The system.

        records:
            The laws of the components that declare and collect without error.

        errors:
            The path of every failing component with its error.

        agents:
            The path of the agent group and the variable of every symbol that reduces agents, an empty variable for
            the symbol of the members.
    """

    system: System
    records: list[CensusRecord]
    errors: list[tuple[str, SkcomplexError]]
    agents: dict[Expression, tuple[str, str]] = field(default_factory=dict)
