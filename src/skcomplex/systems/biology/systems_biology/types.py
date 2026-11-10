"""Define the types of systems biology.

They are the rules of Boolean networks and their candidates.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Iterator, Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

type Rule = Callable[..., bool]
type Candidates = Sequence[tuple[Sequence[int], float]]


class InternalCoordinatesProtocol(Protocol):
    """Internal coordinates of a residue in Biopython.

    The extra `structures` holds Biopython.
    """

    def get_angle(self, name: str) -> float | None:
        """Return the angle of a name in degrees, or `None` when the residue has none.

        Args:
            name:
                The name of the angle.

        Returns:
            angle:
                The angle in degrees, or `None`.
        """

    def set_angle(self, name: str, value: float) -> None:
        """Write the angle of a name in degrees.

        Args:
            name:
                The name of the angle.

            value:
                The angle.
        """


class ResidueProtocol(Protocol):
    """Residue of a chain in Biopython.

    The extra `structures` holds Biopython.
    """

    internal_coord: InternalCoordinatesProtocol


class AtomProtocol(Protocol):
    """Atom of a chain in Biopython.

    The extra `structures` holds Biopython.
    """

    coord: npt.NDArray[np.floating]


class ChainProtocol(Protocol):
    """Chain of a structure in Biopython.

    The extra `structures` holds Biopython. Its `atom_to_internal_coordinates` reads the internal coordinates of the
    positions of the atoms, and its `internal_to_atom_coordinates` writes the positions of the atoms from them.
    """

    atom_to_internal_coordinates: Callable[[], None]
    internal_to_atom_coordinates: Callable[[], None]

    def get_residues(self) -> Iterator[ResidueProtocol]:
        """Return the residues of the chain."""

    def get_atoms(self) -> Iterator[AtomProtocol]:
        """Return the atoms of the chain."""


class BiopythonStructureProtocol(Protocol):
    """Structure in Biopython.

    The extra `structures` holds Biopython.
    """

    def get_chains(self) -> Iterator[ChainProtocol]:
        """Return the chains of the systems of the structure."""
