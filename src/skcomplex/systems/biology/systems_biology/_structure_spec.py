"""Define the spec of a molecular structure."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np

from ....core.errors import ValidationError
from ....core.specs import BaseSpec
from ....core.types import Array

COORDINATE_DIMENSIONS = 3


@dataclass(frozen=True)
class StructureSpec(BaseSpec):
    """Spec of a molecular structure, every system of an ensemble with the same atoms.

    Args:
        coordinates:
            The positions in ångström, systems by atoms by 3.

        atom_names:
            The name of every atom.

        residue_names:
            The name of the residue of every atom.

        residue_numbers:
            The number of the residue of every atom in its chain.

        chain_names:
            The name of the chain of every atom.

    Raises:
        ValidationError:
            The coordinates are not systems by atoms by 3, or the names do not hold one entry per atom.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.systems.biology.systems_biology import StructureSpec
        >>> atoms = StructureSpec(np.zeros((1, 2, 3)), ('N', 'CA'), ('GLY', 'GLY'), (1, 1), ('A', 'A'))
        >>> atoms.residues
        (('A', 1, 'GLY'),)
    """

    coordinates: Array
    atom_names: tuple[str, ...]
    residue_names: tuple[str, ...]
    residue_numbers: tuple[int, ...]
    chain_names: tuple[str, ...]

    def _check(self) -> None:
        """Raise an error for invalid coordinates or names.

        They are invalid when the coordinates have another shape, or the names do not hold one entry per atom.

        Raises:
            ValidationError:
                The coordinates are not systems by atoms by 3, or the names do not hold one entry per atom.
        """
        shape = np.shape(self.coordinates)
        if len(shape) != COORDINATE_DIMENSIONS or shape[-1] != COORDINATE_DIMENSIONS:
            error_msg = f'The coordinates of a structure are systems by atoms by 3, not the shape {shape}.'
            raise ValidationError(error_msg)
        lengths = {len(self.atom_names), len(self.residue_names), len(self.residue_numbers), len(self.chain_names)}
        if lengths != {shape[1]}:
            error_msg = f'The names of a structure hold one entry for each of {shape[1]} atoms, not {sorted(lengths)}.'
            raise ValidationError(error_msg)

    @property
    def residues(self) -> tuple[tuple[str, int, str], ...]:
        """Chain, number, and name of every residue, in the order of the atoms."""
        found = zip(self.chain_names, self.residue_numbers, self.residue_names, strict=True)
        return tuple(dict.fromkeys(found))

    @property
    def effective_atoms(self) -> Array:
        """Mean atom of every residue, systems by residues by 3."""
        keys = list(zip(self.chain_names, self.residue_numbers, self.residue_names, strict=True))
        order = {key: index for index, key in enumerate(dict.fromkeys(keys))}
        members = np.array([order[key] for key in keys])
        return np.stack([self.coordinates[:, members == index].mean(axis=1) for index in range(len(order))], axis=1)
