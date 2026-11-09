"""The fuzzy oil drop of a protein.

The protein is replayed from an ensemble or folded by Monte Carlo moves.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache, partial
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
import rerun as rr
from scipy.optimize import minimize_scalar
from scipy.special import rel_entr

from ....core.components.elements import Element, LatticeElement
from ....core.components.interactions import Interaction, SamplingInteraction
from ....core.components.laws import BaseLaw, DriveLaw
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.specs import CompiledElementSpec
from ....core.components.systems import BaseSystem, RunSettings, System
from ....core.engine.problems import BaseProblemSpec, FrameSpec
from ....core.engine.runtime import Trajectory
from ....core.engine.viz import PointsVisual
from ....core.errors import AnalysisError, ValidationError
from ....core.specs import ObservableSpec, RoleSpec, RowScoreSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..._extras import import_extra
from ._backbone_geometry import BackboneGeometry
from ._structure_spec import StructureSpec
from ._structures import read_structure

CHAIN_DIMENSIONS = 3

KINDS = ('profile', 'relative-distance', 'environment-factor')

SIGMA_FRACTION = 3.0

CUTOFF = 9.0

LARGEST_FACTOR = 5.0

FRAME_DIMENSIONS = 3

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
        >>> from skcomplex.systems.biology.systems_biology import ChainCoordinateElement
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
        >>> from skcomplex.systems.biology.systems_biology import DihedralAngleElement
        >>> DihedralAngleElement('site').angle
        0.0
    """

    def __init__(self, label: str, angle: Annotated[float, StateSpec()] = 0.0) -> None:
        super().__init__(label)
        self.angle = angle


class EnsembleReplayInteraction(Interaction):
    """Drive that sets a lattice state to one member of an ensemble per unit of time.

    It holds the last member after the end.

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
        >>> from skcomplex.systems.biology.systems_biology import EnsembleReplayInteraction
        >>> EnsembleReplayInteraction('replay', target='chain', state='x', ensemble=np.zeros((2, 3))).state
        'x'
    """

    law_classes = (DriveLaw,)

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


class BackboneVisual(PointsVisual):
    """Visual that logs the effective atoms of a chain as points in three dimensions.

    A line strip joins the points in order.

    A state of coordinates holds one row of three values per residue. With a structure, the state holds the angles of a
    `BackboneGeometry` instead, and the visual rebuilds the chain from them at every frame.

    Args:
        position:
            The name of the lattice state of coordinates or angles.

        structure:
            The structure whose chain the angles rebuild, or `None` for a state of coordinates.

        radius:
            The radius of the points, or `None` for a share of the extent of the positions.

        states:
            The names of the other states to log as time series.

        trail:
            The number of recent positions the trail of each point holds, or 0 for no trail.

        strip:
            Whether to join the points in member order by one line strip.

        path:
            The entity path of the group to draw, which also names it in the problem.

        recording:
            The Rerun recording to log to, or `None` for the active recording of the thread.

    Attributes:
        geometry_:
            The chain the angles rebuild, or `None` for a state of coordinates.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import BackboneVisual
        >>> BackboneVisual(position='x').strip
        True
    """

    def __init__(
        self,
        position: str,
        structure: StructureSpec | None = None,
        radius: float | None = None,
        states: Sequence[str] | None = (),
        *,
        trail: int = 0,
        strip: bool = True,
        path: str | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        super().__init__(position, radius, states=states, trail=trail, strip=strip, path=path, recording=recording)
        self.structure = structure

    def _prepare(self, context: BaseProblemSpec | None) -> None:
        """Keep the chain that the angles rebuild, and prepare the visual."""
        self.geometry_ = None if self.structure is None else BackboneGeometry(self.structure).prepare()
        super()._prepare(context)

    def _read_points(self, element: CompiledElementSpec, frame: FrameSpec) -> npt.NDArray[np.float64]:
        """Return the effective atoms at a frame.

        They are rebuilt from the angles when the visual holds a structure.
        """
        values = np.asarray(frame.value(element.states[self.position]), dtype=float)
        if self.geometry_ is None:
            return values.reshape(-1, CHAIN_DIMENSIONS)
        return self.geometry_.run(values.ravel()).read_effective_atoms()

    def _get_dimension(self, element: CompiledElementSpec, problem: BaseProblemSpec) -> tuple[int, ...]:
        """Return the three coordinates of a point."""
        return (CHAIN_DIMENSIONS,)


@dataclass(frozen=True)
class FuzzyOilDropMeasuresResult:
    """Distributions of hydrophobicity of a structure, with their divergences.

    It also holds the values of a kind.

    Args:
        hydrophobicity:
            The hydrophobicity of every residue.

        positions:
            The effective atoms, frames by residues by 3.

        theoretical:
            The distribution `T`, frames by residues.

        observed:
            The distribution `O`, frames by residues.

        uniform:
            The distribution `R`, one value per residue.

        divergences:
            `D_KL(O|T)` and `D_KL(O|R)` in bits, frames by 2.

        values:
            For the kind `'relative-distance'` the `RD`, and for the kind `'environment-factor'` the `K`, per frame, or
            `None` for `'profile'`.
    """

    hydrophobicity: Array
    positions: Array
    theoretical: Array
    observed: Array
    uniform: Array
    divergences: Array
    values: Array | None = None


def fuzzy_oil_drop_measures(
    source: object = None,
    *,
    structure: StructureSpec,
    kind: str = 'relative-distance',
    scale: Mapping[str, float] | None = None,
    cutoff: float = CUTOFF,
    observed: npt.ArrayLike | None = None,
    state: str | None = None,
) -> FuzzyOilDropMeasuresResult:
    """Return the distributions of hydrophobicity of a structure and their relative distance.

    They follow the fuzzy oil drop.

    The theoretical distribution `T` is a three-dimensional Gaussian over the effective atoms along the principal axes
    of the chain (Eq. 1.1). The observed distribution `O` is Levitt's sum over the pairs of residues within a cutoff
    (Eq. 1.2). The uniform distribution `R` gives every residue the same value. Then
    `RD = D_KL(O|T) / (D_KL(O|T) + D_KL(O|R))` in bits (Eqs. 1.3 to 1.6), and `K` minimises `D_KL(O|M)` for the profile
    `M = [T + K [T_max - T]_n]_n` (Eq. 1.9), where `[...]_n` normalises to a sum of one, by SciPy's bounded scalar
    minimiser. The equations follow Konieczny et al. (2023 §1.3.1), and the curves of Fig. 1.41 give back its `K = 0.5`.
    Each sigma is a third of the largest absolute coordinate along its axis, one reading of the book's Gaussian that
    reaches near zero at `3 sigma` (p. 33 and Fig. 1.41), which gives no formula for sigma. The book names no scale, no
    orientation of the axes, no fit of `K`, and no upper bound of it, for which it reports values above 3 (p. 41), so
    the Kyte-Doolittle scale, the principal axes, the minimisation, and the bound 5 are choices of this function, as is
    leaving a residue out of its own sum in Eq. 1.2, where the book speaks of interactions between residues.

    The bundled entries do not reach the `RD < 0.5` and `K` below 0.5 that the book expects of fast-folding and
    single-chain proteins (p. 40): the mean `RD` and `K` are 0.77 and 2.2 for 1L2Y, 0.70 and 0.98 for 1UBQ, and 0.77 and
    2.2 for 5PTI. The gap comes from sigma, since a wider Gaussian closes it and no other choice does.

    Args:
        source:
            A trajectory whose lattice state holds the coordinates, an array of effective atoms, frames by residues by
            3, or `None` for the systems of the structure.

        structure:
            The structure that gives the residues, and the systems when the source is `None`.

        kind:
            `'profile'`, `'relative-distance'`, or `'environment-factor'`.

        scale:
            The hydrophobicity by the three-letter name of a residue, or `None` for the Kyte-Doolittle scale rescaled to
            `[0, 1]`.

        cutoff:
            The cutoff distance of the observed distribution in ångström.

        observed:
            The observed distribution to use in place of Levitt's sum, one value per residue, or `None`.

        state:
            The dotted name of the lattice state of a trajectory that holds the coordinates, or `None` for the first
            state.

    Returns:
        result:
            The distributions, their divergences, and the values of the kind.

    Raises:
        AnalysisError:
            The kind is unknown, the cutoff is not positive, the structure holds a residue the scale does not cover, or
            the positions do not hold one row per residue.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import fuzzy_oil_drop_measures, read_structure
        >>> values = fuzzy_oil_drop_measures(structure=read_structure('1L2Y')).values
        >>> values.shape, bool(values.min() >= 0 and values.max() <= 1)
        ((38,), True)
    """
    if kind not in KINDS:
        error_msg = f'fuzzy_oil_drop_measures has kind={kind!r}. Use one of {", ".join(KINDS)}.'
        raise AnalysisError(error_msg)
    if cutoff <= 0:
        error_msg = f'The cutoff of the observed distribution must be positive, but it is {cutoff}.'
        raise AnalysisError(error_msg)
    hydrophobicity = _read_hydrophobicity(structure, scale)
    positions = _read_positions(source, structure, state)
    count = len(hydrophobicity)
    if positions.ndim != FRAME_DIMENSIONS or positions.shape[1] != count:
        error_msg = f'The positions are frames by {count} residues by 3, not the shape {positions.shape}.'
        raise AnalysisError(error_msg)
    frames = len(positions)
    theoretical = np.array([_read_theoretical(frame) for frame in positions])
    if observed is None:
        found = np.array([_read_observed(frame, hydrophobicity, cutoff) for frame in positions])
    else:
        found = np.tile(np.asarray(observed, dtype=float), (frames, 1))
    uniform = np.full(positions.shape[1], 1 / positions.shape[1])
    divergences = np.array(
        [
            [_read_divergence(row, reference), _read_divergence(row, uniform)]
            for row, reference in zip(found, theoretical, strict=True)
        ],
    )
    total = divergences.sum(axis=1)
    values = None
    if kind == 'relative-distance':
        values = np.divide(divergences[:, 0], total, out=np.zeros(frames), where=total > 0)
    if kind == 'environment-factor':
        values = np.array(
            [_read_environment_factor(row, reference) for row, reference in zip(found, theoretical, strict=True)],
        )
    return FuzzyOilDropMeasuresResult(hydrophobicity, positions, theoretical, found, uniform, divergences, values)


def _read_hydrophobicity(structure: StructureSpec, scale: Mapping[str, float] | None) -> Array:
    """Return the hydrophobicity of every residue.

    It raises an error that names a residue the scale does not cover.
    """
    chosen = read_kyte_doolittle_scale() if scale is None else scale
    name = 'Kyte-Doolittle' if scale is None else 'given'
    for chain, number, residue in structure.residues:
        if residue not in chosen:
            error_msg = (
                f'The residue {residue} at number {number} of chain {chain} is not on the {name} scale, '
                f'which covers {", ".join(sorted(chosen))}. Pass a scale that covers it.'
            )
            raise AnalysisError(error_msg)
    return np.array([chosen[residue] for _, _, residue in structure.residues], dtype=float)


def _read_positions(source: object, structure: StructureSpec, state: str | None) -> Array:
    """Return the effective atoms of the systems.

    They come from the systems, from a lattice state of a trajectory, or from an array.
    """
    if source is None:
        return structure.effective_atoms
    if isinstance(source, Trajectory):
        name = state if state is not None else source.problem.states.names[0]
        return np.asarray(source.lattice(name), dtype=float)
    return np.asarray(source, dtype=float)


def _read_theoretical(positions: Array) -> Array:
    """Return the Gaussian over the positions along their principal axes, normalised."""
    centred = positions - positions.mean(axis=0)
    axes = np.linalg.svd(centred, full_matrices=False)[2]
    aligned = centred @ axes.T
    sigma = np.abs(aligned).max(axis=0) / SIGMA_FRACTION
    values = np.exp(-np.sum(aligned**2 / (2 * sigma**2), axis=1))
    return values / values.sum()


def _read_observed(positions: Array, hydrophobicity: Array, cutoff: float) -> Array:
    """Return Levitt's sum over the pairs of residues within the cutoff, normalised."""
    ratio = np.linalg.norm(positions[:, None] - positions[None], axis=2) / cutoff
    weights = 1 - 0.5 * (7 * ratio**2 - 9 * ratio**4 + 5 * ratio**6 - ratio**8)
    weights[ratio > 1] = 0.0
    np.fill_diagonal(weights, 0.0)
    values = np.sum((hydrophobicity[:, None] + hydrophobicity[None]) * weights, axis=1)
    return values / values.sum()


def _read_divergence(observed: Array, reference: Array) -> float:
    """Return the divergence of a distribution from a reference in bits."""
    return float(np.sum(rel_entr(observed, reference)) / np.log(2))


def _read_environment_factor(observed: Array, theoretical: Array) -> float:
    """Return the factor `K` of the profile closest to the observed distribution."""
    complement = (theoretical.max() - theoretical) / (theoretical.max() - theoretical).sum()

    def divergence(factor: float) -> float:
        """Return the divergence of the observed distribution from the profile of a factor."""
        profile = theoretical + factor * complement
        return _read_divergence(observed, profile / profile.sum())

    return float(minimize_scalar(divergence, bounds=(0.0, LARGEST_FACTOR), method='bounded').x)


def read_kyte_doolittle_scale() -> dict[str, float]:
    """Return the Kyte-Doolittle hydrophobicity of the twenty amino acids of Biopython.

    It is rescaled to `[0, 1]`.

    Returns:
        scale:
            The hydrophobicity by the three-letter name of a residue.

    Raises:
        MissingExtraError:
            The extra `structures` is not installed.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import read_kyte_doolittle_scale
        >>> scale = read_kyte_doolittle_scale()
        >>> scale['ILE'], scale['ARG']
        (1.0, 0.0)
    """
    system = 'fuzzy_oil_drop'
    letters = import_extra('Bio.SeqUtils.ProtParamData', system, 'structures').kd
    names = import_extra('Bio.Data.IUPACData', system, 'structures').protein_letters_1to3
    low, high = min(letters.values()), max(letters.values())
    return {names[letter].upper(): (value - low) / (high - low) for letter, value in letters.items()}


@lru_cache(maxsize=4)
def _read_chain(entry: str) -> tuple[StructureSpec, BackboneGeometry]:
    """Return a bundled entry and its chain.

    The chain rebuilds from angles.
    """
    structure = read_structure(entry)
    return structure, BackboneGeometry(structure).prepare()


def _read_value(structure: StructureSpec, kind: str, positions: Array) -> float:
    """Return a value of the fuzzy oil drop of the effective atoms of a chain."""
    values = fuzzy_oil_drop_measures(positions[None], structure=structure, kind=kind).values
    if values is None:
        error_msg = f'The fuzzy oil drop of the kind {kind!r} has no value per frame.'
        raise AnalysisError(error_msg)
    return float(values[0])


def _read_observed_value(kind: str, entry: str, state: str, t: float, values: Mapping[str, object]) -> float:
    """Return a value of the fuzzy oil drop of a frame.

    The frame holds the coordinates or the angles.
    """
    del t
    structure, geometry = _read_chain(entry)
    numbers = np.asarray(values[state], dtype=float).ravel()
    positions = numbers.reshape(-1, COORDINATES) if state == COORDINATE_STATE else None
    if positions is None:
        positions = geometry.run(numbers).read_effective_atoms()
    return _read_value(structure, kind, positions)


class FuzzyOilDropSystem(BaseSystem):
    """The fuzzy oil drop of a protein.

    It is an ensemble replayed one member per frame or a chain folded by moves.

    The default replays the systems of an entry, which sets the effective atoms of the chain, the mean atom of each
    residue, to one system per unit of time. With `dynamics='folding'`, the chain starts extended, with all `phi =
    -120°` and `psi = 130°`, and each unit of time proposes moves of its dihedral angles, accepted by the row score `RD`
    plus one for every pair of effective atoms closer than 3 Å and at least two residues apart. The observables `RD` and
    `K` follow Konieczny et al. (2023 §1.3.1) through `fuzzy_oil_drop_measures`. The book gives no folding dynamics, so
    the extended start, the clash rule, the rotation width of 60°, and the temperature are choices of this system. The
    system has the lattice `chain` of the state `x` or `angle`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        MissingExtraError:
            The extra `structures` is not installed.

        ValidationError:
            The dynamics are unknown.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import FuzzyOilDropSystem
        >>> FuzzyOilDropSystem().prepare().system_.elements[0].space.shape
        (20, 3)
    """

    field = 'biology/systems_biology'
    topics = ('protein', 'hydrophobicity', 'divergence entropy', 'structures', 'monte carlo')
    references = ('Konieczny et al. 2023', 'Levitt 1976')
    until = 37

    def __init__(
        self,
        entry: str = ENTRY,
        *,
        dynamics: str = 'replay',
        proposals: int = 1,
        width: float = 60.0,
        temperature: float = 0.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.entry = entry
        self.dynamics = dynamics
        self.proposals = proposals
        self.width = width
        self.temperature = temperature

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings.

        They hold the relative distance and the environment factor of the chain.

        Returns:
            settings:
                The tuned run settings by name.
        """
        state = ANGLE_STATE if self.dynamics == 'folding' else COORDINATE_STATE
        settings = super().get_run_settings()
        settings['observables'] = [
            ObservableSpec('RD', function=partial(_read_observed_value, 'relative-distance', self.entry, state)),
            ObservableSpec('K', function=partial(_read_observed_value, 'environment-factor', self.entry, state)),
        ]
        return settings

    @staticmethod
    def _rotate_dihedral(width: float, row: Array, rng: np.random.Generator) -> Array:
        """Return a row of angles with one random angle rotated by up to a width.

        The rotated angle is wrapped to `[-180, 180)`.
        """
        site = rng.integers(row.size)
        row[site] = (row[site] + rng.uniform(-width, width) + 180.0) % 360.0 - 180.0
        return row

    @staticmethod
    def _read_clashes(positions: Array) -> int:
        """Return the number of clashes of effective atoms.

        A clash is a pair of atoms apart along the chain and closer than the clash distance.
        """
        rows, columns = np.triu_indices(len(positions), NEAREST_SEPARATED_RESIDUES)
        return int(np.sum(np.linalg.norm(positions[rows] - positions[columns], axis=1) < CLASH_DISTANCE))

    def _score_angles(self, entry: str, row: Array) -> float:
        """Return the relative distance of a chain rebuilt from a row of angles.

        It adds one for every clash.
        """
        structure, geometry = _read_chain(entry)
        positions = geometry.run(row).read_effective_atoms()
        return _read_value(structure, 'relative-distance', positions) + self._read_clashes(positions)

    def _build_system(self) -> System:
        """Return the parts of the fuzzy oil drop of a protein."""
        import_extra('Bio.PDB', 'fuzzy_oil_drop', 'structures')
        if self.dynamics not in DYNAMICS:
            error_msg = f'Unknown dynamics {self.dynamics!r}. Use one of {", ".join(DYNAMICS)}.'
            raise ValidationError(error_msg)
        structure = read_structure(self.entry)
        if self.dynamics == 'replay':
            ensemble = structure.effective_atoms
            return System(
                self.label,
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
        angles = np.full(BackboneGeometry(structure).prepare().angles_.size, EXTENDED_PHI)
        angles[angles.size // 2 :] = EXTENDED_PSI
        return System(
            self.label,
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
                    objective=RowScoreSpec(partial(self._score_angles, self.entry), proposals=self.proposals),
                    move=partial(self._rotate_dihedral, self.width),
                    intensity=float('inf') if self.temperature == 0 else 1 / self.temperature,
                ),
            ],
            time=DiscreteTime(),
        )
