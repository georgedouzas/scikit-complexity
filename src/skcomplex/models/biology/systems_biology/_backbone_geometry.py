"""Rebuild a protein chain from its backbone dihedral angles through the internal coordinates of Biopython."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings

import numpy as np

from ....base import BaseConfig
from ....errors import ValidationError
from ....types import Array
from ._structure_spec import StructureSpec
from ._structures import read_biopython_structure

VECTOR_WARNING = "'where' used without 'out'"
VECTOR_MODULE = r'Bio\.PDB\.vectors'


class BackboneGeometry(BaseConfig[None, Array]):
    """Chain of a structure rebuilt from the angles `phi` and `psi` of its backbone, with all other geometry kept.

    The angles of a chain of `n` residues are one array of `2 (n - 1)` values in degrees: `phi` of the residues from the
    second to the last, then `psi` of the residues from the first to the one before the last. The bond lengths, the
    bond angles, `omega`, and the side chains stay as they are in the model. Biopython warns on every rebuild that NumPy
    reads `where` without `out`, and the class filters this one warning.

    Args:
        structure:
            The structure, of one chain.

        model:
            The index of the model that gives the geometry.

    Attributes:
        angles_:
            The angles of the model, in degrees.

        positions_:
            The positions of the atoms after the last run, atoms by 3.

    Raises:
        ValidationError:
            The structure holds more than one chain, or the angles have another length.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.models.biology.systems_biology import BackboneGeometry, read_structure
        >>> geometry = BackboneGeometry(read_structure('1L2Y')).check().prepare()
        >>> geometry.angles_.shape
        (38,)
        >>> geometry.run(geometry.angles_).positions_.shape
        (304, 3)
    """

    def __init__(self, structure: StructureSpec, model: int = 0) -> None:
        self.structure = structure
        self.model = model

    def _check(self) -> None:
        """Raise an error for a structure of more than one chain."""
        chains = set(self.structure.chain_names)
        if len(chains) != 1:
            error_msg = f'A backbone is rebuilt for one chain, but the structure holds the chains {sorted(chains)}.'
            raise ValidationError(error_msg)

    def _rebuild(self) -> None:
        """Rebuild the atoms from the internal coordinates, filtering the one warning of Biopython's vectors."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=VECTOR_WARNING, category=UserWarning, module=VECTOR_MODULE)
            self._chain.internal_to_atom_coordinates()

    def _prepare(self, context: None = None) -> None:
        """Read the internal coordinates of the chain, and set the angles of the model."""
        self._chain = next(read_biopython_structure(self.structure, (self.model,)).get_chains())
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=VECTOR_WARNING, category=UserWarning, module=VECTOR_MODULE)
            self._chain.atom_to_internal_coordinates()
        self._residues = [residue.internal_coord for residue in self._chain.get_residues()]
        phi = [residue.get_angle('phi') for residue in self._residues[1:]]
        psi = [residue.get_angle('psi') for residue in self._residues[:-1]]
        self.angles_ = np.array(phi + psi, dtype=float)

    def _run(self, inputs: Array | None) -> None:
        """Write the angles into the chain, and set the positions of the rebuilt atoms."""
        angles = self.angles_ if inputs is None else np.asarray(inputs, dtype=float)
        if angles.shape != self.angles_.shape:
            error_msg = f'A backbone reads {self.angles_.size} angles, phi then psi, not the shape {angles.shape}.'
            raise ValidationError(error_msg)
        split = len(self._residues) - 1
        for residue, angle in zip(self._residues[1:], angles[:split], strict=True):
            residue.set_angle('phi', float(angle))
        for residue, angle in zip(self._residues[:-1], angles[split:], strict=True):
            residue.set_angle('psi', float(angle))
        self._rebuild()
        self.positions_ = np.array([atom.coord for atom in self._chain.get_atoms()], dtype=float)

    def read_effective_atoms(self) -> Array:
        """Return the mean atom of every residue of the last run.

        Returns:
            positions:
                Residues by 3.

        Examples:
            >>> from skcomplex.models.biology.systems_biology import BackboneGeometry, read_structure
            >>> geometry = BackboneGeometry(read_structure('1L2Y')).check().prepare()
            >>> geometry.run(geometry.angles_).read_effective_atoms().shape
            (20, 3)
        """
        names = self.structure
        return StructureSpec(
            self.positions_[None],
            names.atom_names,
            names.residue_names,
            names.residue_numbers,
            names.chain_names,
        ).effective_atoms[0]
