"""Describe the rows of a variable group, which members take and free as a run adds and removes them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..base import CapacityError, VariableKind, declarations_of
from ._arrays import Indices, Vector
from ._declarations import ACTIVE, ENDS, Block, Declarations
from ._layout import Entry
from ._neighbourhoods import ACTIVE_THRESHOLD
from ._tree import coerce_value


@dataclass(frozen=True)
class CapacityRows:
    """Rows of a variable group on the nodes or edges of a network, with the values a new member starts from.

    Args:
        path:
            The path of the group.

        variables:
            The entry of every state and parameter of the members by name, with whether it is a state.

        defaults:
            The flat values of one new member of every variable by name.

        active:
            The parameter entry of the activity of every row.

        source:
            The parameter entry of the source node of every edge row, or `None` for a group on the nodes.

        target:
            The parameter entry of the target node of every edge row, or `None` for a group on the nodes.
    """

    path: str
    variables: Mapping[str, tuple[Entry, bool]]
    defaults: Mapping[str, Vector]
    active: Entry
    source: Entry | None = None
    target: Entry | None = None

    def present(self, p: Vector) -> npt.NDArray[np.bool_]:
        """Return whether every row holds a member.

        Args:
            p:
                The parameter vector.

        Returns:
            present:
                One flag per row.
        """
        return self.active.view(p)[:, 0] > ACTIVE_THRESHOLD

    def take_row(self, p: Vector) -> int:
        """Return the first free row.

        Args:
            p:
                The parameter vector.

        Returns:
            row:
                The row.

        Raises:
            CapacityError:
                Every row holds a member.
        """
        free = np.flatnonzero(~self.present(p))
        if not free.size:
            error_msg = (
                f'The group {self.path} holds {self.active.rows} members, its capacity, and a topology request adds '
                'another. Give the group a larger capacity, `Group(..., variable=True, capacity=n)`.'
            )
            raise CapacityError(error_msg)
        return int(free[0])

    def write(self, row: int, x: Vector, p: Vector, values: Mapping[str, object]) -> None:
        """Start a member in a row with given values or the defaults, and mark the row present.

        Args:
            row:
                The row.

            x:
                The state vector, changed in place.

            p:
                The parameter vector, changed in place.

            values:
                The values of the new member by variable name, which override the defaults.
        """
        for name, (entry, is_state) in self.variables.items():
            vector = x if is_state else p
            value = values.get(name, self.defaults[name])
            start = entry.start + row * entry.columns
            vector[start : start + entry.columns] = np.broadcast_to(np.asarray(value, dtype=float), (entry.columns,))
        p[self.active.start + row] = 1.0

    def clear(self, row: int, p: Vector) -> None:
        """Mark a row free.

        Args:
            row:
                The row.

            p:
                The parameter vector, changed in place.
        """
        p[self.active.start + row] = 0.0

    def ends(self, p: Vector) -> tuple[Indices, Indices]:
        """Return the source and target node of every edge row.

        Args:
            p:
                The parameter vector.

        Returns:
            ends:
                The source and target node rows.
        """
        assert self.source is not None  # nosec B101: called for edge groups only
        assert self.target is not None  # nosec B101: called for edge groups only
        return self.source.view(p)[:, 0].astype(np.intp), self.target.view(p)[:, 0].astype(np.intp)


def capacity_rows(block: Block, declarations: Declarations) -> CapacityRows:
    """Return the rows of a variable group with the values of one new member.

    Args:
        block:
            The block of the variable group.

        declarations:
            The declarations of the compilation.

    Returns:
        rows:
            The capacity rows of the group.
    """
    node = block.node
    element = node.component_type
    states = declarations_of(element, VariableKind.STATE)
    variables: dict[str, tuple[Entry, bool]] = {}
    defaults: dict[str, Vector] = {}
    for name, entry in block.entries.items():
        if name in {ACTIVE, *ENDS}:
            continue
        variables[name] = (entry, name in states)
        default = element.get_default(name) if name in element.declarations() else 0.0
        default = 0.0 if default is dataclasses.MISSING else default
        defaults[name] = coerce_value(entry.name, default, entry.member_shape, entry.domain, None)
    source, target = block.entries.get(ENDS[0]), block.entries.get(ENDS[1])
    return CapacityRows(node.path, variables, defaults, block.entries[ACTIVE], source, target)
