"""Define the census spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass, field

from sage.all__sagemath_symbolics import Expression

from ...errors import SkcomplexError
from ...specs._base_spec import BaseSpec
from ..system import System
from ._census_record_spec import CensusRecordSpec


@dataclass(frozen=True)
class CensusSpec(BaseSpec):
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
    records: list[CensusRecordSpec]
    errors: list[tuple[str, SkcomplexError]]
    agents: dict[Expression, tuple[str, str]] = field(default_factory=dict)
