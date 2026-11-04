"""Systems biology model: the fuzzy oil drop of a protein, replayed from an ensemble or folded by Monte Carlo moves."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from functools import lru_cache, partial
from types import SimpleNamespace
from typing import Annotated

import numpy as np

from ....core.elements import Element, LatticeElement
from ....core.interactions import Interaction, SamplingInteraction
from ....core.laws import BaseLaw, DriveLaw
from ....core.spaces import DiscreteTime, LatticeSpace
from ....core.system import System
from ....errors import ValidationError
from ....specs import ObservableSpec, RoleSpec, RowScoreSpec, StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model
from ..._extras import import_extra
from ._backbone_geometry import BackboneGeometry
from ._backbone_visual import BackboneVisual
from ._fuzzy_oil_drop_analysis import FuzzyOilDropAnalysis
from ._structure_spec import StructureSpec
from ._structures import read_structure

ENTRY = '1L2Y'
DYNAMICS = ('replay', 'folding')
COORDINATES = 3
EXTENDED_PHI = -120.0
EXTENDED_PSI = 130.0
CLASH_DISTANCE = 3.0
NEAREST_SEPARATED_RESIDUES = 2
COORDINATE_STATE = 'chain.x'
ANGLE_STATE = 'chain.angle'


class ChainCoordinateElement(Element):
    """Coordinate of an effective atom of a protein chain.

    Args:
        label:
            The label of the element.

        x:
            The coordinate in ångström.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import ChainCoordinateElement
        >>> ChainCoordinateElement('site').x
        0.0
    """

    def __init__(self, label: str, x: Annotated[float, StateSpec()] = 0.0) -> None:
        super().__init__(label)
        self.x = x


class DihedralAngleElement(Element):
    """Backbone dihedral angle of a protein chain.

    Args:
        label:
            The label of the element.

        angle:
            The angle in degrees.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import DihedralAngleElement
        >>> DihedralAngleElement('site').angle
        0.0
    """

    def __init__(self, label: str, angle: Annotated[float, StateSpec()] = 0.0) -> None:
        super().__init__(label)
        self.angle = angle


class EnsembleReplayInteraction(Interaction):
    """Drive that sets a lattice state to one member of an ensemble per unit of time, holding the last after the end.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the lattice group.

        state:
            The name of the state.

        ensemble:
            The members, members by the lattice shape.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.models.biology.systems_biology import EnsembleReplayInteraction
        >>> EnsembleReplayInteraction('replay', target='chain', state='x', ensemble=np.zeros((2, 3))).state
        'x'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str,
        ensemble: Array,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.ensemble = ensemble

    def _read_member(self, values: Array, *, time: float) -> Array:
        """Return the member of the time of a step."""
        return self.ensemble[min(int(time), len(self.ensemble) - 1)].reshape(values.shape)

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseLaw]:
        """Return the drive of the state."""
        return [DriveLaw(getattr(self._get_member(namespace, 'target'), self.state), self._read_member)]


@lru_cache(maxsize=4)
def _read_chain(entry: str) -> tuple[StructureSpec, BackboneGeometry]:
    """Return a bundled entry and its chain, which rebuilds from angles."""
    structure = read_structure(entry)
    return structure, BackboneGeometry(structure).check().prepare()


def _rotate_dihedral(width: float, row: Array, random: np.random.Generator) -> Array:
    """Return a row of angles with one random angle rotated by up to a width and wrapped to `[-180, 180)`."""
    site = random.integers(row.size)
    row[site] = (row[site] + random.uniform(-width, width) + 180.0) % 360.0 - 180.0
    return row


def _read_clashes(positions: Array) -> int:
    """Return the number of pairs of effective atoms apart along the chain and closer than the clash distance."""
    rows, columns = np.triu_indices(len(positions), NEAREST_SEPARATED_RESIDUES)
    return int(np.sum(np.linalg.norm(positions[rows] - positions[columns], axis=1) < CLASH_DISTANCE))


def _read_value(structure: StructureSpec, kind: str, positions: Array) -> float:
    """Return a value of the fuzzy oil drop of the effective atoms of a chain."""
    return float(FuzzyOilDropAnalysis(structure, kind).check().prepare(positions[None]).run().values_[0])


def _score_angles(entry: str, row: Array) -> float:
    """Return the relative distance of a chain rebuilt from a row of angles, plus one for every clash."""
    structure, geometry = _read_chain(entry)
    positions = geometry.run(row).read_effective_atoms()
    return _read_value(structure, 'relative-distance', positions) + _read_clashes(positions)


def _read_observed_value(kind: str, entry: str, state: str, t: float, values: Mapping[str, object]) -> float:
    """Return a value of the fuzzy oil drop of the coordinates or the angles that a frame holds."""
    del t
    structure, geometry = _read_chain(entry)
    numbers = np.asarray(values[state], dtype=float).ravel()
    positions = numbers.reshape(-1, COORDINATES) if state == COORDINATE_STATE else None
    if positions is None:
        positions = geometry.run(numbers).read_effective_atoms()
    return _read_value(structure, kind, positions)


def _read_observables(system: System) -> list[ObservableSpec]:
    """Return the relative distance and the environment factor of the chain of a system, labelled by its entry."""
    state = COORDINATE_STATE if isinstance(system.interactions[0], EnsembleReplayInteraction) else ANGLE_STATE
    return [
        ObservableSpec('RD', function=partial(_read_observed_value, 'relative-distance', system.label, state)),
        ObservableSpec('K', function=partial(_read_observed_value, 'environment-factor', system.label, state)),
    ]


@register_model(
    field='biology/systems_biology',
    tags=('protein', 'hydrophobicity', 'divergence entropy', 'structures', 'monte carlo'),
    references=('Konieczny et al. 2023', 'Levitt 1976'),
    defaults={'until': 37},
    observables=_read_observables,
)
def fuzzy_oil_drop(
    entry: str = ENTRY,
    *,
    dynamics: str = 'replay',
    proposals: int = 1,
    width: float = 60.0,
    temperature: float = 0.0,
) -> System:
    """Return the fuzzy oil drop of a protein, an ensemble replayed one member per frame or a chain folded by moves.

    The default replays the models of an entry, which sets the effective atoms of the chain, the mean atom of each
    residue, to one model per unit of time. With `dynamics='folding'`, the chain starts extended, with all `phi = -120°`
    and `psi = 130°`, and each unit of time proposes moves of its dihedral angles, accepted by the row score `RD` plus
    one for every pair of effective atoms closer than 3 Å and at least two residues apart. The observables `RD` and `K`
    follow Konieczny et al. (2023 §1.3.1) through `FuzzyOilDropAnalysis`.

    Args:
        entry:
            The bundled entry, `'1L2Y'`, `'1UBQ'`, or `'5PTI'`.

        dynamics:
            `'replay'` or `'folding'`.

        proposals:
            The number of dihedral moves per unit of time of the folding.

        width:
            The largest rotation of a dihedral move, in degrees.

        temperature:
            The temperature of the folding, one over the intensity of its sampling, where 0 accepts only moves that do
            not raise the score.

    Returns:
        system:
            The model, with the lattice `chain` of the state `x` or `angle`.

    Raises:
        MissingExtraError:
            The extra `structures` is not installed.

        ValidationError:
            The dynamics are unknown.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import fuzzy_oil_drop
        >>> fuzzy_oil_drop().elements[0].space.shape
        (20, 3)
    """
    import_extra('Bio.PDB', 'fuzzy_oil_drop', 'structures')
    if dynamics not in DYNAMICS:
        error_msg = f'Unknown dynamics {dynamics!r}. Use one of {", ".join(DYNAMICS)}.'
        raise ValidationError(error_msg)
    structure = read_structure(entry)
    if dynamics == 'replay':
        ensemble = structure.effective_atoms
        return System(
            label=entry,
            elements=[
                LatticeElement(
                    'chain',
                    element=ChainCoordinateElement,
                    space=LatticeSpace((ensemble.shape[1], COORDINATES), boundary='fixed'),
                    values={'x': ensemble[0]},
                    visual=VisualSpec(BackboneVisual, {'position': 'x'}),
                ),
            ],
            interactions=[EnsembleReplayInteraction('replay', target='chain', state='x', ensemble=ensemble)],
            time=DiscreteTime(),
        )
    angles = np.full(BackboneGeometry(structure).check().prepare().angles_.size, EXTENDED_PHI)
    angles[angles.size // 2 :] = EXTENDED_PSI
    return System(
        label=entry,
        elements=[
            LatticeElement(
                'chain',
                element=DihedralAngleElement,
                space=LatticeSpace((1, angles.size), boundary='fixed'),
                values={'angle': angles[None]},
                visual=VisualSpec(BackboneVisual, {'position': 'angle', 'structure': structure}),
            ),
        ],
        interactions=[
            SamplingInteraction(
                'folding',
                target='chain',
                state='angle',
                objective=RowScoreSpec(partial(_score_angles, entry), proposals=proposals),
                move=partial(_rotate_dihedral, width),
                intensity=float('inf') if temperature == 0 else 1 / temperature,
            ),
        ],
        time=DiscreteTime(),
    )
