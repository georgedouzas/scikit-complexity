"""Define the base of the neighbourhoods of the members of a space."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from ..base import BasePart


class BaseNeighborhood(BasePart[None, SimpleNamespace]):
    """Base of the neighbourhoods, the parts of a system that give the members a law reads around a member of a space.

    A neighbourhood is a part: it stores its arguments in its own `__init__`, `check` validates them, and `prepare`
    derives what the engine reads, such as the offsets of a lattice neighbourhood. The space that reads it rejects a
    neighbourhood of another kind. It contributes no laws.

    Args:
        label:
            The label of the neighbourhood.
    """

    def _run(self, inputs: SimpleNamespace | None) -> None:
        """Run nothing, since a neighbourhood contributes no laws to a system."""

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return no laws, since a neighbourhood only tells laws which members they read."""
        return []

    def _check(self) -> None:
        """Accept the neighbourhood, which has nothing to check besides its label by default."""

    def _prepare(self, context: None = None) -> None:
        """Leave the neighbourhood as it is, since it has nothing to prepare by default."""
