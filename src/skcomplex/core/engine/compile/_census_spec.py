"""Define the spec of the laws of every component of a system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from sage.all__sagemath_symbolics import Expression

from ...base import BaseSpec
from ...errors import SkcomplexError
from ..problems._composite_system import _CompositeSystem
from ._census_record_spec import _CensusRecordSpec


@dataclass(frozen=True)
class _CensusSpec(BaseSpec):
    """A spec of the laws and the errors of every component of a system.

    A component fails when it cannot declare or collect its laws, and the census keeps its error instead of its laws.

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

    system: _CompositeSystem
    records: list[_CensusRecordSpec]
    errors: list[tuple[str, SkcomplexError]]
    agents: Mapping[Expression, tuple[str, str]] = field(default_factory=dict)
