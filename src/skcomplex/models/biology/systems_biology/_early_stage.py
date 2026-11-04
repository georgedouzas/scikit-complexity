"""Systems biology model: the early-stage folding intermediate, residues sampled on the cells of an ellipse path."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import lru_cache, partial
from typing import Annotated

import numpy as np

from ....core.elements import Element, LatticeElement
from ....core.interactions import SamplingInteraction
from ....core.spaces import DiscreteTime, LatticeSpace
from ....core.system import System
from ....errors import ValidationError
from ....specs import ParameterSpec, RowScoreSpec, StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model
from ..._extras import import_extra
from ._backbone_geometry import BackboneGeometry
from ._ellipse_path import EllipsePath
from ._structure_spec import StructureSpec
from ._structures import BUNDLED, read_structure

AMINO_ACIDS = (
    'ALA',
    'ARG',
    'ASN',
    'ASP',
    'CYS',
    'GLN',
    'GLU',
    'GLY',
    'HIS',
    'ILE',
    'LEU',
    'LYS',
    'MET',
    'PHE',
    'PRO',
    'SER',
    'THR',
    'TRP',
    'TYR',
    'VAL',
)
ENTRY = '5PTI'
CELLS = 72


def read_native_pairs(structure: StructureSpec) -> tuple[list[str], Array]:
    """Return the residues with both a `phi` and a `psi` in the first model of a structure, and these angles.

    Args:
        structure:
            The structure, of one chain.

    Returns:
        pairs:
            The names of the residues, and their `phi` and `psi` in radians, residues by 2.

    Raises:
        MissingExtraError:
            The extra `structures` is not installed.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import read_native_pairs, read_structure
        >>> names, angles = read_native_pairs(read_structure('1L2Y'))
        >>> len(names), angles.shape
        (18, (18, 2))
    """
    angles = BackboneGeometry(structure).check().prepare().angles_
    split = angles.size // 2
    names = [residue for _, _, residue in structure.residues][1:-1]
    pairs = np.column_stack([angles[: split - 1], angles[split + 1 :]])
    return names, np.radians(pairs)


@lru_cache(maxsize=4)
def read_cell_probabilities(cells: int = CELLS) -> Array:
    """Return the probability of every cell of the ellipse path per amino acid, from the native bundled structures.

    The projected native angles of the bundled entries are counted per amino acid and cell, with one added to every
    count. The bundled entries and the added one are choices of this function, since Konieczny et al. (2023 §3.7.1)
    print entropies of such probabilities (Table 3.3) and not the probabilities or how they are estimated.

    Args:
        cells:
            The number of cells of the path.

    Returns:
        probabilities:
            One row per amino acid, in the order of `AMINO_ACIDS`, and one column per cell.

    Raises:
        MissingExtraError:
            The extra `structures` is not installed.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import read_cell_probabilities
        >>> read_cell_probabilities().sum(axis=1).round(9).tolist() == [1.0] * 20
        True
    """
    counts = np.ones((len(AMINO_ACIDS), cells))
    path = EllipsePath(cells=cells).check().prepare()
    for entry in BUNDLED:
        names, pairs = read_native_pairs(read_structure(entry))
        for name, cell in zip(names, path.run(pairs).cells_, strict=True):
            counts[AMINO_ACIDS.index(name), cell] += 1
    return counts / counts.sum(axis=1, keepdims=True)


class PathCellElement(Element):
    """Residue of a protein chain whose dihedral angles lie in one cell of an ellipse path.

    Args:
        label:
            The label of the element.

        cell:
            The cell of the path.

        amino:
            The index of the amino acid of the residue.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import PathCellElement
        >>> PathCellElement('residue').cell
        0.0
    """

    def __init__(
        self,
        label: str,
        cell: Annotated[float, StateSpec()] = 0.0,
        amino: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.cell = cell
        self.amino = amino


def _score_cells(energies: Array, row: Array, amino: Array) -> float:
    """Return the sum of the local energies of the cells of the residues of a chain."""
    return float(energies[amino.astype(int), row.astype(int)].sum())


@register_model(
    field='biology/systems_biology',
    tags=('protein', 'ramachandran', 'early stage', 'heat bath', 'monte carlo'),
    references=('Konieczny et al. 2023', 'Brylinski et al. 2004', 'Brylinski et al. 2005'),
    defaults={'until': 300},
)
def early_stage_folding_intermediate(
    entry: str = ENTRY,
    *,
    cells: int = CELLS,
    temperature: float = 1.0,
) -> System:
    """Return the residues of a protein sampling the cells of an ellipse path of the Ramachandran map by heat bath.

    Each residue takes a cell of `EllipsePath`, the limited conformational space of the early stage after Konieczny et
    al. (2023 §3.7.1), with the local energy `-log p(cell | amino acid)` of the probabilities of
    `read_cell_probabilities`. The heat bath and the 72 cells of equal `t` are choices of this model: the 72 of the book
    counts 5° steps of one angle of the whole map (p. 147).
    Each unit of time makes as many draws as the chain has residues, each drawing the cell of one random residue from
    `p(cell | amino acid)` raised to one over the temperature, renormalised, so at temperature 1 the cells follow the
    probabilities. The local energies enter through a row score, their sum over the chain, since the change of the score
    with the cell of one residue is the change of its local energy.

    Args:
        entry:
            The bundled entry whose sequence the model samples.

        cells:
            The number of cells of the path.

        temperature:
            The temperature, in the units of the local energy.

    Returns:
        system:
            The model, with the lattice `residues` of the state `cell`.

    Raises:
        MissingExtraError:
            The extra `structures` is not installed.

        ValidationError:
            The temperature is not positive.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import early_stage_folding_intermediate
        >>> early_stage_folding_intermediate().elements[0].space.shape
        (1, 58)
    """
    import_extra('Bio.PDB', 'early_stage_folding_intermediate', 'structures')
    if temperature <= 0:
        error_msg = f'The temperature of the model must be positive, but it is {temperature}.'
        raise ValidationError(error_msg)
    names = [residue for _, _, residue in read_structure(entry).residues]
    amino = np.array([AMINO_ACIDS.index(name) for name in names], dtype=float)
    energies = -np.log(read_cell_probabilities(cells))
    return System(
        elements=[
            LatticeElement(
                'residues',
                element=PathCellElement,
                space=LatticeSpace((1, len(names)), boundary='fixed'),
                values={'amino': amino[None]},
                visual=VisualSpec('image', {'states': ('cell',), 'low': 0.0, 'high': float(cells - 1)}),
            ),
        ],
        interactions=[
            SamplingInteraction(
                'sampling',
                target='residues',
                state='cell',
                objective=RowScoreSpec(partial(_score_cells, energies), proposals=len(names)),
                move='random-value',
                intensity=1 / temperature,
                rule='heat-bath',
                values=tuple(float(cell) for cell in range(cells)),
            ),
        ],
        time=DiscreteTime(),
    )
