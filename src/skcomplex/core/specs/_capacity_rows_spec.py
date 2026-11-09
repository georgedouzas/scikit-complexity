"""Define the capacity rows spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..errors import CapacityError
from ..types import Array, Indices
from ._base_spec import BaseSpec
from ._entry_spec import EntrySpec

ACTIVE_THRESHOLD = 0.5


@dataclass(frozen=True)
class CapacityRowsSpec(BaseSpec):
    """Rows of a variable group on a network, with the values of a new member.

    The group is on the nodes, edges, or hyperedges of the network.

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

        members:
            The parameter entry of the member nodes of every hyperedge row, one column per member, or `None` for a
            group on the nodes or edges.
    """

    path: str
    variables: Mapping[str, tuple[EntrySpec, bool]]
    defaults: Mapping[str, Array]
    active: EntrySpec
    source: EntrySpec | None = None
    target: EntrySpec | None = None
    members: EntrySpec | None = None

    def get_present(self, p: Array) -> npt.NDArray[np.bool_]:
        """Return whether every row holds a member.

        Args:
            p:
                The parameter vector.

        Returns:
            present:
                One flag per row.
        """
        return self.active.get_view(p)[:, 0] > ACTIVE_THRESHOLD

    def take_row(self, p: Array) -> int:
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
        free = np.flatnonzero(~self.get_present(p))
        if not free.size:
            error_msg = (
                f'The group {self.path} holds {self.active.rows} members, its capacity, and a topology request adds '
                'another. Give the group a larger capacity, `GroupElement(..., variable=True, capacity=n)`.'
            )
            raise CapacityError(error_msg)
        return int(free[0])

    def get_ends(self, p: Array) -> tuple[Indices, Indices]:
        """Return the source and target node of every edge row.

        Args:
            p:
                The parameter vector.

        Returns:
            ends:
                The source and target node rows.
        """
        assert self.source is not None
        assert self.target is not None
        return self.source.get_view(p)[:, 0].astype(np.intp), self.target.get_view(p)[:, 0].astype(np.intp)

    def get_hyperedges(self, p: Array) -> Indices:
        """Return the member nodes of every hyperedge row.

        Args:
            p:
                The parameter vector.

        Returns:
            members:
                One row of member nodes per hyperedge row.
        """
        assert self.members is not None
        return self.members.get_view(p).astype(np.intp)
