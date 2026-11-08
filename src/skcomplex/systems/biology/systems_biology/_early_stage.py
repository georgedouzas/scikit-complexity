"""The early-stage folding intermediate of a protein.

Its residues are sampled on the cells of an ellipse path.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import lru_cache, partial
from typing import Annotated

import numpy as np
from scipy.optimize import minimize_scalar

from ....core.base import BaseConfig
from ....core.components.elements import Element, LatticeElement
from ....core.components.interactions import SamplingInteraction
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.system import BaseSystem, System
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RowScoreSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..._extras import import_extra
from ._backbone_geometry import BackboneGeometry
from ._structure_spec import StructureSpec
from ._structures import BUNDLED, read_structure

LONG_DIAGONAL = np.radians(83.0)

SHORT_DIAGONAL = np.radians(58.0)

GRID_POINTS = 3600

TURN = 2 * np.pi

PAIR_DIMENSIONS = 2

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


class PathCellElement(Element):
    """Residue of a protein chain with dihedral angles in one cell of an ellipse path.

    Args:
        label:
            The label of the element.

        cell:
            The cell of the path.

        amino:
            The index of the amino acid of the residue.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import PathCellElement
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


class EllipsePath(BaseConfig[None, Array]):
    """Ellipse path on the Ramachandran map.

    It projects dihedral angles on it by the shortest periodic distance.

    The path is `phi = A cos(t) - B sin(t)`, `psi = -A cos(t) - B sin(t)`, a centred ellipse whose long diagonal runs
    from the upper left to the lower right of the map, with `t = 0` in the lower right and `t` growing clockwise.
    Konieczny et al. (2023 §3.7.1, Fig. 3.43d and e) draw this path and print no equation or number for it. The
    defaults, 83° and 58°, are readings of a figure and not published numbers: a fit to panel e of that figure gives
    84.5° and 56.3° with the centre at the origin, which agrees within the 2.7° of one of its pixels. The origin
    and the direction of `t` are a convention of this class, since the book states neither. The equation and the
    parameters of Brylinski et al. (2004, Bioinformatics 20:199) have not been read.

    Args:
        long_diagonal:
            The value `A` in radians.

        short_diagonal:
            The value `B` in radians.

        cells:
            The number of equal cells of the parameter `t`.

    Attributes:
        parameters_:
            The parameter `t` of the projection of every pair of the last run, in `[0, 2 pi)`.

        angles_:
            The projected `phi` and `psi` of the last run in radians, pairs by 2.

        cells_:
            The cell of the path of every pair of the last run.

    Raises:
        ValidationError:
            A diagonal or the number of cells is not positive, or the angles are not pairs.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.systems.biology.systems_biology import EllipsePath
        >>> path = EllipsePath().prepare()
        >>> path.read_points(np.array([0.0])).shape
        (1, 2)
        >>> path.run(np.array([[1.0, -1.0]])).cells_.tolist()
        [0]
    """

    def __init__(
        self,
        long_diagonal: float = LONG_DIAGONAL,
        short_diagonal: float = SHORT_DIAGONAL,
        cells: int = 72,
    ) -> None:
        self.long_diagonal = long_diagonal
        self.short_diagonal = short_diagonal
        self.cells = cells

    def _prepare(self, context: None = None) -> None:
        """Set the grid of the path that starts the search of a projection.

        It first raises an error for a diagonal or a number of cells that is not positive.
        """
        if self.long_diagonal <= 0 or self.short_diagonal <= 0 or self.cells < 1:
            error_msg = (
                'The diagonals and the cells of an ellipse path must be positive, but they are '
                f'{self.long_diagonal}, {self.short_diagonal}, and {self.cells}.'
            )
            raise ValidationError(error_msg)
        self.grid_ = np.linspace(0.0, TURN, GRID_POINTS, endpoint=False)
        self.grid_points_ = self.read_points(self.grid_)

    def read_points(self, parameters: Array) -> Array:
        """Return the points of the path at some values of its parameter.

        Args:
            parameters:
                The values of `t`.

        Returns:
            points:
                The `phi` and `psi` in radians, values by 2.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.systems.biology.systems_biology import EllipsePath
            >>> phi, psi = EllipsePath(1.0, 0.5).read_points(np.array([0.0]))[0]
            >>> float(phi), float(psi)
            (1.0, -1.0)
        """
        along, across = self.long_diagonal * np.cos(parameters), self.short_diagonal * np.sin(parameters)
        return np.column_stack([along - across, -along - across])

    def _read_distance(self, parameter: float, pair: Array) -> float:
        """Return the shortest periodic distance between a pair of angles and the path.

        The distance is to a point of the path.
        """
        point = self.read_points(np.array([parameter]))[0]
        difference = (pair - point + np.pi) % TURN - np.pi
        return float(np.hypot(*difference))

    def _project(self, pair: Array) -> float:
        """Return the parameter of the point of the path closest to a pair.

        It refines the best points of the grid.
        """
        differences = (pair - self.grid_points_ + np.pi) % TURN - np.pi
        distances = np.hypot(differences[:, 0], differences[:, 1])
        best = float(self.grid_[np.argmin(distances)])
        step = TURN / GRID_POINTS
        refined = minimize_scalar(
            self._read_distance,
            bounds=(best - step, best + step),
            args=(pair,),
            method='bounded',
            options={'xatol': 1e-10},
        )
        return float(refined.x if refined.fun <= distances.min() else best) % TURN

    def _run(self, inputs: Array | None) -> None:
        """Set the projection of pairs of `phi` and `psi` in radians."""
        pairs = np.asarray(inputs, dtype=float)
        if pairs.ndim != PAIR_DIMENSIONS or pairs.shape[1] != PAIR_DIMENSIONS:
            error_msg = f'An ellipse path projects pairs of phi and psi, not the shape {pairs.shape}.'
            raise ValidationError(error_msg)
        self.parameters_ = np.array([self._project(pair) for pair in pairs])
        self.angles_ = self.read_points(self.parameters_)
        self.cells_ = np.minimum((self.parameters_ / TURN * self.cells).astype(int), self.cells - 1)


def read_native_pairs(structure: StructureSpec) -> tuple[list[str], Array]:
    """Return the residues with both a `phi` and a `psi`, and these angles.

    The residues are those of the first system of a structure.

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
        >>> from skcomplex.systems.biology.systems_biology import read_native_pairs, read_structure
        >>> names, angles = read_native_pairs(read_structure('1L2Y'))
        >>> len(names), angles.shape
        (18, (18, 2))
    """
    angles = BackboneGeometry(structure).prepare().angles_
    split = angles.size // 2
    names = [residue for _, _, residue in structure.residues][1:-1]
    pairs = np.column_stack([angles[: split - 1], angles[split + 1 :]])
    return names, np.radians(pairs)


@lru_cache(maxsize=4)
def read_cell_probabilities(cells: int = CELLS) -> Array:
    """Return the probability of every cell of the ellipse path per amino acid.

    It reads the native bundled structures.

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
        >>> from skcomplex.systems.biology.systems_biology import read_cell_probabilities
        >>> read_cell_probabilities().sum(axis=1).round(9).tolist() == [1.0] * 20
        True
    """
    counts = np.ones((len(AMINO_ACIDS), cells))
    path = EllipsePath(cells=cells).prepare()
    for entry in BUNDLED:
        names, pairs = read_native_pairs(read_structure(entry))
        for name, cell in zip(names, path.run(pairs).cells_, strict=True):
            counts[AMINO_ACIDS.index(name), cell] += 1
    return counts / counts.sum(axis=1, keepdims=True)


class EarlyStageFoldingIntermediateSystem(BaseSystem):
    """The residues of a protein sampling the cells of an ellipse path by heat bath.

    The path lies on the Ramachandran map.

    Each residue takes a cell of `EllipsePath`, the limited conformational space of the early stage after Konieczny et
    al. (2023 §3.7.1), with the local energy `-log p(cell | amino acid)` of the probabilities of
    `read_cell_probabilities`. The heat bath and the 72 cells of equal `t` are choices of this system: the 72 of the
    book counts 5° steps of one angle of the whole map (p. 147). Each unit of time makes as many draws as the chain has
    residues, each drawing the cell of one random residue from `p(cell | amino acid)` raised to one over the
    temperature, renormalised, so at temperature 1 the cells follow the probabilities. The local energies enter through
    a row score, their sum over the chain, since the change of the score with the cell of one residue is the change of
    its local energy. The system has the lattice `residues` of the state `cell`.

    Args:
        entry:
            The bundled entry whose sequence the system samples.

        cells:
            The number of cells of the path.

        temperature:
            The temperature, in the units of the local energy.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        MissingExtraError:
            The extra `structures` is not installed.

        ValidationError:
            The temperature is not positive.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import EarlyStageFoldingIntermediateSystem
        >>> EarlyStageFoldingIntermediateSystem().prepare().system_.elements[0].space.shape
        (1, 58)
    """

    field = 'biology/systems_biology'
    topics = ('protein', 'ramachandran', 'early stage', 'heat bath', 'monte carlo')
    references = ('Konieczny et al. 2023', 'Brylinski et al. 2004', 'Brylinski et al. 2005')
    until = 300

    def __init__(
        self,
        entry: str = ENTRY,
        *,
        cells: int = CELLS,
        temperature: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.entry = entry
        self.cells = cells
        self.temperature = temperature

    @staticmethod
    def _score_cells(energies: Array, row: Array, amino: Array) -> float:
        """Return the sum of the local energies of the cells of the residues of a chain."""
        return float(energies[amino.astype(int), row.astype(int)].sum())

    def _build_system(self) -> System:
        """Return the parts of the system."""
        import_extra('Bio.PDB', 'early_stage_folding_intermediate', 'structures')
        if self.temperature <= 0:
            error_msg = f'The temperature of the system must be positive, but it is {self.temperature}.'
            raise ValidationError(error_msg)
        names = [residue for _, _, residue in read_structure(self.entry).residues]
        amino = np.array([AMINO_ACIDS.index(name) for name in names], dtype=float)
        energies = -np.log(read_cell_probabilities(self.cells))
        return System(
            self.label,
            elements=[
                LatticeElement(
                    'residues',
                    element=PathCellElement,
                    space=LatticeSpace((1, len(names)), boundary='fixed'),
                    values={'amino': amino[None]},
                    visual=VisualSpec('image', {'states': ('cell',), 'low': 0.0, 'high': float(self.cells - 1)}),
                ),
            ],
            interactions=[
                SamplingInteraction(
                    'sampling',
                    target='residues',
                    state='cell',
                    objective=RowScoreSpec(partial(self._score_cells, energies), proposals=len(names)),
                    move='random-value',
                    intensity=1 / self.temperature,
                    rule='heat-bath',
                    values=tuple(float(cell) for cell in range(self.cells)),
                ),
            ],
            time=DiscreteTime(),
        )
