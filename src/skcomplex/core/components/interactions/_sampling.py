"""Define Monte Carlo sampling of a state, by moves accepted from the change of a local objective."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

import networkx as nx
import numba
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import RR, SR, ConditionSet, Expression

from ...errors import ValidationError
from ...specs import (
    HolonomyScoreSpec,
    MatrixGroupSpec,
    MoveSpec,
    ObjectiveSpec,
    ParameterSpec,
    RegionScoreSpec,
    RoleSpec,
    RowScoreSpec,
    SamplerSpec,
    Tags,
)
from ...types import Array, ComplexArray, ElementKind, Indices
from ..laws import BaseLaw, DriveLaw, SampleLaw
from ..spaces import (
    AdjacencyNeighborhood,
    BaseNeighborhood,
    BaseSpace,
    CellComplexSpace,
    LatticeSpace,
    NetworkSpace,
    VonNeumannNeighborhood,
)
from ._interaction import Interaction

MOVES = ('flip', 'random-value', 'rotate', 'gaussian', 'cluster', 'copy')
HOLONOMY_MOVES = ('rotate', 'random-value')
HOLONOMY_SCHEDULES = ('colored', 'random-sequential')
COLORING_STRATEGY = 'largest_first'
MATRIX_AXES = 2


def _is_positive(value: float) -> bool:
    """Return whether a value is above zero."""
    return bool(value > 0)


POSITIVE_VALUES = ConditionSet(RR, _is_positive)


@numba.njit(cache=True)
def _elastic_change(strength: float, target: float, total: float, step: float) -> float:  # pragma: no cover
    """Return the change of `strength * (total - target)^2` when the total grows by a step."""
    return strength * step * (2.0 * (total - target) + step)


@numba.njit(cache=True)
def _copy_labels(
    labels: Indices,
    contacts: tuple[Indices, Indices, Array, Array],
    scores: tuple[Array, Array, Array],
    proposals: tuple[Indices, Array, Array],
    acceptance: tuple[float, bool],
) -> None:  # pragma: no cover
    """Propose label copies at the given sites, accept them in place, and keep the totals of every label.

    Every attempt copies the label of a neighbour of a site, the neighbour at the fraction `picks` of its neighbours,
    into the site. The contacts are the neighbours of every site, their number, the weights of the contacts, and the
    volumes of the sites. The change of the objective sums the contact energies `contact` between labels weighted by the
    contacts of the site, and the change of the elastic constraints of the two labels, of strengths `constraints[0]` and
    targets `constraints[1]` on the volumes `totals[0]`, and of strengths `constraints[2]` and targets `constraints[3]`
    on the surfaces `totals[1]`. An accepted copy updates the totals of the two labels only. The proposals are the
    sites, the picks, and the uniform numbers of the acceptance, and the acceptance the intensity and whether the rule
    is the Metropolis rule.
    """
    neighbors, counts, weights, sizes = contacts
    contact, constraints, totals = scores
    sites, picks, draws = proposals
    intensity, metropolis = acceptance
    for attempt in range(sites.size):
        site = sites[attempt]
        count = counts[site]
        if count == 0:
            continue
        old = labels[site]
        new = labels[neighbors[site, min(int(picks[attempt] * count), count - 1)]]
        if old == new:
            continue
        change = 0.0
        old_surface = 0.0
        new_surface = 0.0
        for slot in range(count):
            other = labels[neighbors[site, slot]]
            weight = weights[site, slot]
            if other == new:
                new_surface -= weight
            else:
                change += weight * contact[new, other]
                new_surface += weight
            if other == old:
                old_surface += weight
            else:
                change -= weight * contact[old, other]
                old_surface -= weight
        size = sizes[site]
        change += _elastic_change(constraints[0, old], constraints[1, old], totals[0, old], -size)
        change += _elastic_change(constraints[0, new], constraints[1, new], totals[0, new], size)
        change += _elastic_change(constraints[2, old], constraints[3, old], totals[1, old], old_surface)
        change += _elastic_change(constraints[2, new], constraints[3, new], totals[1, new], new_surface)
        exponent = 0.0 if change == 0.0 else change * intensity
        if metropolis:
            accepted = exponent <= 0.0 or draws[attempt] < math.exp(-exponent)
        else:
            accepted = draws[attempt] < 1.0 / (1.0 + math.exp(min(exponent, 700.0)))
        if accepted:
            labels[site] = new
            totals[0, old] -= size
            totals[0, new] += size
            totals[1, old] += old_surface
            totals[1, new] += new_surface


class SamplingInteraction(Interaction):
    """Monte Carlo sweeps of a state of the members of a lattice or network group, one sweep per unit of time.

    Every sweep proposes a move of the state of every member and accepts it from the change of the local objective of
    the member, the terms of the objective that involve its value, at the intensity. The local objective is a function
    of the namespace, which holds `namespace.target`, the states and parameters of the member, and
    `namespace.target.other`, a neighbour, whose values `neighbors(...)` sums over `neighborhood`. A move is `'flip'`,
    the negative value, `'random-value'`, a value drawn from `values`, `'rotate'`, the value plus a uniform number in
    `[-width, width]`, `'gaussian'`, the value plus a normal number of standard deviation `width`, or a function
    `move(values, random)` of the array of the current values of some members and the random generator that returns
    their proposed values. The move `'cluster'` is the Wolff algorithm instead: a sweep flips clusters until the flipped
    members number as many as the members, each cluster grown from a random member through the bonds to neighbours
    with probability `1 - exp(min(0, -intensity * (f(-s, n) - f(s, n))))`, where `f(s, n)` is the term of the local
    objective of a member of value `s` with a neighbour of value `n`. It takes an objective of such pair terms that the
    flip of both values keeps, and ignores the rule and the schedule. A group on the edges of a cell complex takes a
    local objective that sums over `CofaceNeighborhood`, inside which `exterior_derivative` of the state is the oriented
    sum around the one face, and its `'colored'` schedule updates together only links that share no face.

    An objective that is a `RowScoreSpec` scores whole rows of a state on a `LatticeSpace` instead, along the last axis.
    Every unit of time proposes its `proposals` moves per row, each accepted from the change of the score of the row,
    and a run records the score of every row as `<label>.score`. A named move changes one random site of the row, a
    function move receives the row, the heat-bath rule draws the value of one random site among `values` by the weights
    `exp(-intensity * score)`, and the schedule and the neighbourhood do not apply. An infinite intensity accepts
    a move that keeps the objective with probability 1 under the Metropolis rule and 1/2 under the Glauber rule, and it
    never accepts a move that raises the objective.

    An objective that is a `HolonomyScoreSpec` samples a matrix group state on the edges of a `CellComplexSpace`
    instead, whose local objective of an edge sums a class function over the holonomies of the faces around it. The
    move `'rotate'` multiplies the matrix of an edge from the left by a random matrix of the group near the identity of
    spread `width`, `'random-value'` draws it from the Haar measure of the group, and a function move receives the
    matrices of some edges and the random generator and returns their proposed matrices. Every sweep proposes `hits`
    moves per edge, the `'colored'` schedule updates together only edges that share no face, and the matrices are
    projected back onto the group at the end of every sweep, which undoes the rounding of repeated products.

    An objective that is a `RegionScoreSpec` samples a state of integer labels from 0 on the sites of a `LatticeSpace`
    or the nodes of a `NetworkSpace` instead, whose sites of one label form a region, by the move `'copy'`. Every sweep
    proposes `hits` copies per site, each at a random site, of the label of one of its neighbours drawn at random, and
    accepts it from the change of the contact energy between regions and of the volume and surface constraints of the
    two regions it changes. The sampler keeps the volume and the surface of every region and updates them on every
    accepted copy. The neighbourhood is that of the copies and the contacts, by default the Moore neighbourhood of a
    square lattice, the six neighbours of a triangular lattice, or the edges of a network, whose edge attribute
    `'weight'` weighs the contact and node attribute `'volume'` the volume of a site. The schedule does not apply.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group on the sites of a lattice, the nodes of a network, or the edges of a cell complex.

        state:
            The name of the state the moves change.

        objective:
            The function `objective(namespace)` of the local objective of a member, a `RowScoreSpec`, a
            `HolonomyScoreSpec`, or a `RegionScoreSpec`.

        move:
            `'flip'`, `'random-value'`, `'rotate'`, `'gaussian'`, `'cluster'`, `'copy'` of the label of a neighbour,
            which a region score takes, or a function that proposes values.

        intensity:
            The factor of the change of the objective in the acceptance `exp(-intensity * change)`, positive and
            possibly infinite.

        rule:
            The acceptance, `'metropolis'`, `'glauber'`, or `'heat-bath'`, which draws among all values of a discrete
            move by the weights `exp(-intensity * objective)`.

        schedule:
            `'checkerboard'`, `'colored'`, or `'random-sequential'`.

        neighborhood:
            The neighbourhood the objective reduces over, or `None` for the von Neumann neighbourhood of a square
            lattice. A network takes `AdjacencyNeighborhood()`.

        values:
            The values a random-value move draws from.

        width:
            The largest change of a rotation, or the standard deviation of a Gaussian move.

        cooling:
            The function of the time symbol that multiplies one over the intensity, or `None` for a constant intensity.

    Raises:
        ValidationError:
            The move, the rule, or the schedule is unknown, the objective is not callable, or a row score takes a
            cluster move, or the heat-bath rule without a random-value move and its values, or a holonomy score takes a
            move, a rule, or a schedule other than those of a matrix group, or a region score takes a move other than
            `'copy'` or the heat-bath rule, or the move `'copy'` has an objective other than a region score.

    Examples:
        >>> from skcomplex.core.components.elements import Element, LatticeElement
        >>> from skcomplex.core.components.interactions import SamplingInteraction
        >>> from skcomplex.core.components.laws.utils import neighbors
        >>> from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.core.specs import StateSpec
        >>> from typing import Annotated
        >>> class SpinElement(Element):
        ...     def __init__(self, label, s: Annotated[float, StateSpec()] = 1.0):
        ...         super().__init__(label)
        ...         self.s = s
        >>> objective = lambda n: -n.target.s * neighbors(n.target.other.s)
        >>> ising = SamplingInteraction('ising', target='spins', state='s', objective=objective, intensity=1.0)
        >>> spins = LatticeElement('spins', element=SpinElement, space=LatticeSpace((8, 8)))
        >>> system = System(elements=[spins], interactions=[ising], time=DiscreteTime())
        >>> states = Simulation(system, random_state=0).prepare().run(until=20).trajectory_.states
        >>> bool(abs(states[-1].mean()) > 0.9)
        True
        >>> from sage.all__sagemath_symbolics import cos
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.core.components.laws.utils import exterior_derivative
        >>> from skcomplex.core.components.spaces import CellComplexSpace, CofaceNeighborhood
        >>> class LinkElement(Element):
        ...     def __init__(self, label, theta: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label)
        ...         self.theta = theta
        >>> wilson = lambda n: -neighbors(cos(exterior_derivative(n.target.theta)), over=CofaceNeighborhood())
        >>> gauge = SamplingInteraction(
        ...     'wilson', target='links', state='theta', objective=wilson, move='rotate', schedule='colored'
        ... )
        >>> links = GroupElement('links', element=LinkElement, space=CellComplexSpace(LatticeSpace((4, 4))), on='edges')
        >>> system = System(elements=[links], interactions=[gauge], time=DiscreteTime())
        >>> Simulation(system, random_state=0).prepare().run(until=2).trajectory_.states.shape
        (3, 32)
        >>> import numpy as np
        >>> from skcomplex.core.specs import RowScoreSpec
        >>> def flip_one(row, random):
        ...     row[random.integers(row.size)] *= -1
        ...     return row
        >>> walk = SamplingInteraction(
        ...     'walk', target='spins', state='s', objective=RowScoreSpec(np.sum), move=flip_one, intensity=float('inf')
        ... )
        >>> spins = LatticeElement('spins', element=SpinElement, space=LatticeSpace((2, 8)))
        >>> system = System(elements=[spins], interactions=[walk], time=DiscreteTime())
        >>> Simulation(system, random_state=0).prepare().run(until=3).trajectory_.values('walk.score').shape
        (4, 2)
        >>> from skcomplex.core.specs import HolonomyScoreSpec, MatrixGroupSpec
        >>> group = MatrixGroupSpec('SU', 2)
        >>> class MatrixLinkElement(Element):
        ...     def __init__(self, label, U: Annotated[object, StateSpec(group=group)] = tuple(group.identity)):
        ...         super().__init__(label)
        ...         self.U = U
        >>> trace = lambda holonomies: -np.trace(holonomies, axis1=-2, axis2=-1).real
        >>> score = HolonomyScoreSpec(trace, group)
        >>> wilson = SamplingInteraction(
        ...     'wilson', target='links', state='U', objective=score, move='rotate', width=0.5, schedule='colored'
        ... )
        >>> space = CellComplexSpace(LatticeSpace((4, 4)))
        >>> links = GroupElement('links', element=MatrixLinkElement, space=space, on='edges')
        >>> system = System(elements=[links], interactions=[wilson], time=DiscreteTime())
        >>> Simulation(system, random_state=0).prepare().run(until=2).trajectory_.values('links.U').shape
        (3, 32, 8)
        >>> from skcomplex.core.specs import RegionScoreSpec
        >>> class SiteElement(Element):
        ...     def __init__(self, label, region: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label)
        ...         self.region = region
        >>> contact = lambda first, second: np.ones(np.broadcast(first, second).shape)
        >>> area = lambda labels: (1.0, 16.0)
        >>> copies = SamplingInteraction(
        ...     'copies', target='sites', state='region', objective=RegionScoreSpec(contact, area), move='copy'
        ... )
        >>> regions = np.kron(np.arange(4).reshape(2, 2), np.ones((4, 4)))
        >>> sites = LatticeElement('sites', element=SiteElement, space=LatticeSpace((8, 8)), values={'region': regions})
        >>> system = System(elements=[sites], interactions=[copies], time=DiscreteTime())
        >>> final = Simulation(system, random_state=0).prepare().run(until=20).state_
        >>> sorted(np.unique(final).tolist())
        [0.0, 1.0, 2.0, 3.0]
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (DriveLaw, SampleLaw)

    def _get_tags(self) -> Tags:
        """Return the tags of a Monte Carlo sweep of a function, or of the drives of a declared score, which draw."""
        sweep = callable(self._get_argument('objective'))
        return self._get_variant_tags((SampleLaw,) if sweep else (DriveLaw,), stochastic=True)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('state',))],
        state: str,
        objective: Callable[[SimpleNamespace], object] | RowScoreSpec | HolonomyScoreSpec | RegionScoreSpec,
        move: str | Callable[..., object] = 'flip',
        intensity: Annotated[float, ParameterSpec(domain=POSITIVE_VALUES)] = 1.0,
        rule: str = 'metropolis',
        schedule: str = 'checkerboard',
        neighborhood: BaseNeighborhood | None = None,
        values: Sequence[float] = (),
        width: float = 1.0,
        cooling: Callable[[Expression], object] | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.objective = objective
        self.move = move
        self.intensity = intensity
        self.rule = rule
        self.schedule = schedule
        self.neighborhood = neighborhood
        self.values = values
        self.width = width
        self.cooling = cooling

    def _get_move(self, target: object) -> MoveSpec:
        """Return the spec of the move of a state."""
        if callable(self.move):
            return MoveSpec(target, 'function', function=self.move)
        return MoveSpec(target, self.move, tuple(self.values), self.width)

    def _get_intensity(self, namespace: SimpleNamespace) -> object:
        """Return the intensity of the acceptance before cooling."""
        return self._get_parameter_symbol('intensity', namespace)

    def _check_row_score(self, score: RowScoreSpec) -> None:
        """Raise an error for a row score that cannot run, a cluster move, or a heat-bath rule without values."""
        score._validated()
        if self.move == 'cluster':
            error_msg = f'The row score of sampling {self.label!r} accepts single moves of a row, not cluster moves.'
            raise ValidationError(error_msg)
        if self.rule == 'heat-bath' and (self.move != 'random-value' or not len(self.values)):
            error_msg = (
                f'The row score of sampling {self.label!r} draws by heat bath among values. Give the move '
                "'random-value' and its `values`."
            )
            raise ValidationError(error_msg)

    def _get_row_keywords(self, score: RowScoreSpec, index: int, parameters: Mapping[str, object]) -> dict[str, object]:
        """Return the parameters that a row score names, those of one value per site read at the row."""
        try:
            names = inspect.signature(score.function).parameters
        except (TypeError, ValueError):
            return {}
        return {
            name: value[index] if isinstance(value, np.ndarray) and value.ndim > 1 else value
            for name, value in parameters.items()
            if name in names
        }

    def _score_rows(self, values: Array, **parameters: object) -> npt.NDArray[np.float64]:
        """Return the score of every row of a state."""
        score = getattr(self, self._get_parameter('objective'))
        rows = values.reshape(-1, values.shape[-1])
        return np.array(
            [
                float(score.function(row, **self._get_row_keywords(score, index, parameters)))
                for index, row in enumerate(rows)
            ],
        )

    def _draw_row_value(
        self,
        row: Array,
        score: RowScoreSpec,
        keywords: Mapping[str, object],
        intensity: float,
        rng: np.random.Generator,
    ) -> Array:
        """Return a row with the value of one random site drawn among all values by the weights of their scores."""
        candidates = np.asarray(self.values, dtype=float)
        proposal = row.copy()
        site = rng.integers(row.size)
        scores = np.empty(candidates.size)
        for position, value in enumerate(candidates):
            proposal[site] = value
            scores[position] = float(score.function(proposal, **keywords))
        excess = scores - scores.min()
        weights = np.exp(-np.where(excess == 0.0, 0.0, excess * np.where(excess == 0.0, 1.0, intensity)))
        proposal[site] = rng.choice(candidates, p=weights / weights.sum())
        return proposal

    def _propose_row(self, row: Array, rng: np.random.Generator) -> Array:
        """Return a row after one move: a function of the row, or a named move at one random site."""
        if callable(self.move):
            return np.asarray(self.move(row.copy(), rng), dtype=float)
        proposal = row.copy()
        site = rng.integers(row.size)
        if self.move == 'flip':
            proposal[site] = -row[site]
        elif self.move == 'random-value':
            proposal[site] = rng.choice(np.asarray(self.values, dtype=float))
        elif self.move == 'gaussian':
            proposal[site] = row[site] + rng.normal(0.0, self.width)
        else:
            proposal[site] = row[site] + rng.uniform(-self.width, self.width)
        return proposal

    def _accept_change(self, change: float, intensity: float, rng: np.random.Generator) -> bool:
        """Return whether a change of the score is accepted by the rule at the intensity."""
        exponent = 0.0 if change == 0.0 else change * intensity
        if self.rule == 'metropolis':
            return exponent <= 0.0 or bool(rng.random() < np.exp(-exponent))
        with np.errstate(over='ignore'):
            return bool(rng.random() < 1.0 / (1.0 + np.exp(exponent)))

    def _accept_rows(self, values: Array, *, random: np.random.Generator, time: float, **parameters: object) -> Array:
        """Return a state after the proposed moves of every row, each accepted from the change of the row's score."""
        score = getattr(self, self._get_parameter('objective'))
        intensity = float(SR(self._get_intensity(SimpleNamespace(**parameters))))
        if self.cooling is not None:
            intensity /= float(SR(self.cooling(time)))
        rows = values.reshape(-1, values.shape[-1]).copy()
        for index in range(len(rows)):
            keywords = self._get_row_keywords(score, index, parameters)
            if self.rule == 'heat-bath':
                for _ in range(score.proposals):
                    rows[index] = self._draw_row_value(rows[index], score, keywords, intensity, random)
                continue
            current = float(score.function(rows[index], **keywords))
            for _ in range(score.proposals):
                proposal = self._propose_row(rows[index], random)
                proposed = float(score.function(proposal, **keywords))
                if self._accept_change(proposed - current, intensity, random):
                    rows[index], current = proposal, proposed
        return rows.reshape(values.shape)

    def _check_holonomy_score(self, score: HolonomyScoreSpec) -> None:
        """Raise an error for a holonomy score with a move, a rule, or a schedule that a matrix group does not take."""
        score._validated()
        if not callable(self.move) and self.move not in HOLONOMY_MOVES:
            error_msg = (
                f'The holonomy score of sampling {self.label!r} moves matrices of a group. Use the move '
                f'{" or ".join(map(repr, HOLONOMY_MOVES))}, or a function of the matrices and the random generator.'
            )
            raise ValidationError(error_msg)
        if self.rule == 'heat-bath':
            error_msg = f"The holonomy score of sampling {self.label!r} takes the rule 'metropolis' or 'glauber'."
            raise ValidationError(error_msg)
        if self.schedule not in HOLONOMY_SCHEDULES:
            error_msg = (
                f'The holonomy score of sampling {self.label!r} updates edges of a cell complex. Use the schedule '
                f'{" or ".join(map(repr, HOLONOMY_SCHEDULES))}.'
            )
            raise ValidationError(error_msg)

    @staticmethod
    @functools.cache
    def _get_staples(space: CellComplexSpace) -> tuple[tuple[Indices, ...], Indices, Array, Array, Array]:
        """Return the tables of the faces around every edge of a cell complex, read once per complex.

        Args:
            space:
                The prepared cell complex.

        Returns:
            classes:
                The classes of a proper coloring of the graph that joins the edges that share a face.

            edges:
                For every edge and face around it, the other edges of the face in path order after the edge.

            signs:
                The sign of the traversal of every edge of `edges`, 0 for padding.

            orientations:
                The sign of the traversal of the edge itself in every face around it.

            real:
                Whether every face around an edge is a face of the complex and not padding.
        """
        cofaces, boundaries, turns = space.cofaces_[1], space.boundaries_, space.boundary_signs_
        real = cofaces < boundaries.shape[0]
        faces = np.where(real, cofaces, 0)
        around, signs = boundaries[faces], turns[faces]
        lengths = (signs != 0).sum(axis=-1)
        position = np.argmax(around == np.arange(cofaces.shape[0])[:, None, None], axis=-1)
        orientations = np.take_along_axis(signs, position[..., None], axis=-1)[..., 0]
        steps = np.arange(max(boundaries.shape[1] - 1, 0))
        after = (position[..., None] + 1 + steps) % np.maximum(lengths, 1)[..., None]
        kept = (steps < lengths[..., None] - 1) & real[..., None]
        edges = np.where(kept, np.take_along_axis(around, after, axis=-1), 0)
        shared = (abs(space.incidences_[1]).T @ abs(space.incidences_[1])).tocoo()
        graph = nx.Graph()
        graph.add_nodes_from(range(cofaces.shape[0]))
        graph.add_edges_from(zip(shared.row.tolist(), shared.col.tolist(), strict=True))
        graph.remove_edges_from(nx.selfloop_edges(graph))
        colors = nx.greedy_color(graph, strategy=COLORING_STRATEGY)
        found = np.asarray([colors[edge] for edge in range(cofaces.shape[0])], dtype=np.intp)
        classes = tuple(np.flatnonzero(found == color) for color in range(int(found.max(initial=-1)) + 1))
        return classes, edges, np.where(kept, np.take_along_axis(signs, after, axis=-1), 0.0), orientations, real

    def _get_local_scores(
        self,
        matrices: ComplexArray,
        staples: ComplexArray,
        orientations: Array,
        real: Array,
        keywords: Mapping[str, object],
    ) -> Array:
        """Return the local objective of every edge, the score of the holonomies of its faces that start at it."""
        score = getattr(self, self._get_parameter('objective'))
        inverse = np.conj(np.swapaxes(matrices, -1, -2))
        oriented = np.where((orientations > 0)[..., None, None], matrices[:, None], inverse[:, None])
        values = np.asarray(score.function(oriented @ staples, **keywords), dtype=float)
        return np.where(real, values, 0.0).sum(axis=-1)

    def _propose_matrices(
        self,
        matrices: ComplexArray,
        group: MatrixGroupSpec,
        rng: np.random.Generator,
    ) -> ComplexArray:
        """Return the proposed matrices of some edges by the move."""
        if callable(self.move):
            return np.asarray(self.move(matrices.copy(), rng), dtype=complex)
        if self.move == 'random-value':
            return group.uniform_of(rng, len(matrices))
        return group.near_identity_of(rng, self.width, len(matrices)) @ matrices

    def _accept_changes(self, changes: Array, intensity: float, rng: np.random.Generator) -> npt.NDArray[np.bool_]:
        """Return whether every change of an objective is accepted by the rule at the intensity."""
        exponents = np.where(changes == 0.0, 0.0, changes * np.where(changes == 0.0, 1.0, intensity))
        draws = rng.random(changes.shape)
        with np.errstate(over='ignore'):
            if self.rule == 'metropolis':
                return (exponents <= 0.0) | (draws < np.exp(-np.maximum(exponents, 0.0)))
            return draws < 1.0 / (1.0 + np.exp(exponents))

    def _sample_holonomies(
        self,
        values: Array,
        *,
        random: np.random.Generator,
        time: float,
        space: BaseSpace,
        **parameters: object,
    ) -> Array:
        """Return a matrix group state on the edges of a cell complex after one sweep of moves of every edge."""
        score = getattr(self, self._get_parameter('objective'))
        if (
            not isinstance(space, CellComplexSpace)
            or values.ndim != MATRIX_AXES
            or values.shape[0] != len(space.cofaces_[1])
        ):
            error_msg = (
                f'The holonomy score of sampling {self.label!r} reads a group on the edges of a CellComplexSpace.'
            )
            raise ValidationError(error_msg)
        intensity = float(SR(self._get_intensity(SimpleNamespace(**parameters))))
        if self.cooling is not None:
            intensity /= float(SR(self.cooling(time)))
        group = score.group
        matrices = group.matrices_of(values)
        classes, edges, signs, orientations, real = self._get_staples(space)
        keywords = self._get_keywords(score.function, parameters)
        order = classes if self.schedule == 'colored' else tuple(random.permutation(values.shape[0])[:, None])
        for links in order:
            local = {
                name: (
                    value[links][:, None]
                    if isinstance(value, np.ndarray) and value.shape[:1] == values.shape[:1]
                    else value
                )
                for name, value in keywords.items()
            }
            staples = group.holonomy_of(matrices, edges[links], signs[links])
            current = self._get_local_scores(matrices[links], staples, orientations[links], real[links], local)
            for _ in range(score.hits):
                proposal = self._propose_matrices(matrices[links], group, random)
                proposed = self._get_local_scores(proposal, staples, orientations[links], real[links], local)
                accepted = self._accept_changes(proposed - current, intensity, random)
                matrices[links[accepted]] = proposal[accepted]
                current = np.where(accepted, proposed, current)
        return group.components_of(group.projection_of(matrices))

    def _check_region_score(self, score: RegionScoreSpec) -> None:
        """Raise an error for a region score with a move other than a label copy, or the heat-bath rule."""
        score._validated()
        if self.move != 'copy':
            error_msg = f"The region score of sampling {self.label!r} moves labels by the move 'copy'."
            raise ValidationError(error_msg)
        if self.rule == 'heat-bath':
            error_msg = f"The region score of sampling {self.label!r} takes the rule 'metropolis' or 'glauber'."
            raise ValidationError(error_msg)

    @staticmethod
    @functools.cache
    def _get_contacts(space: BaseSpace, neighborhood: BaseNeighborhood | None) -> tuple[Indices, Indices, Array, Array]:
        """Return the tables of the neighbours of every site, the weights of their contacts, and the site volumes.

        A lattice site has the neighbours of the neighbourhood, or of the default neighbourhood of the lattice, with
        contacts of weight 1 and a volume of 1. A node of a network has its neighbours along the edges of the layer of
        the neighbourhood, with the edge attribute `'weight'` of the contact and the node attribute `'volume'`, both 1
        by default.

        Args:
            space:
                The prepared lattice or network.

            neighborhood:
                The neighbourhood of the contacts, or `None` for the default neighbourhood of the space.

        Returns:
            neighbors:
                The index of every neighbour of every site, the other sites first, padded with the site itself.

            counts:
                The number of neighbours of every site.

            weights:
                The weight of the contact of every site with every neighbour, 0 for padding.

            sizes:
                The volume of every site.
        """
        over = space._default_neighborhood() if neighborhood is None else neighborhood
        size = space.size_
        sizes = np.ones(size)
        if isinstance(space, NetworkSpace):
            layer = over.layer if isinstance(over, AdjacencyNeighborhood) else None
            graph = space.graphs_[layer]
            indptr, others, edges = space.adjacency_[layer]
            rows = np.repeat(np.arange(size), np.diff(indptr))
            edge_weights = np.asarray([data.get('weight', 1.0) for *_, data in graph.edges(data=True)], dtype=float)
            nodes = space.graph.nodes(data='volume', default=1.0)
            sizes = np.asarray([volume for _, volume in nodes], dtype=float)
            table = np.full((size, int(np.diff(indptr).max(initial=0))), size, dtype=np.intp)
            slots = np.arange(others.size) - indptr[rows]
            table[rows, slots] = others
            values = np.zeros(table.shape)
            values[rows, slots] = edge_weights[edges] if edges.size else 0.0
        else:
            table = space._index_table(over)
            values = np.ones(table.shape)
        sites = np.arange(size)[:, None]
        valid = (table < size) & (table != sites)
        order = np.argsort(~valid, axis=1, kind='stable')
        valid = np.take_along_axis(valid, order, axis=1)
        neighbors = np.where(valid, np.take_along_axis(table, order, axis=1), sites)
        weights = np.where(valid, np.take_along_axis(values, order, axis=1), 0.0)
        return neighbors, valid.sum(axis=1).astype(np.intp), weights, sizes

    def _get_constraint(
        self,
        function: Callable[..., object] | None,
        labels: Indices,
        parameters: Mapping[str, object],
    ) -> tuple[Array, Array]:
        """Return the strength and the target of a constraint of every label, zero for no constraint."""
        if function is None:
            return np.zeros(labels.size), np.zeros(labels.size)
        constraint = np.asarray(function(labels, **self._get_keywords(function, parameters)), dtype=object)
        strengths = np.asarray(np.broadcast_to(np.asarray(constraint[0], dtype=np.float64), labels.shape), dtype=float)
        targets = np.asarray(np.broadcast_to(np.asarray(constraint[1], dtype=np.float64), labels.shape), dtype=float)
        return strengths, targets

    def _sample_regions(
        self,
        values: Array,
        *,
        random: np.random.Generator,
        time: float,
        space: BaseSpace,
        **parameters: object,
    ) -> Array:
        """Return a state of labels after one sweep of label copies, as many as `hits` times the sites."""
        rng = random
        score = getattr(self, self._get_parameter('objective'))
        if not isinstance(space, (LatticeSpace, NetworkSpace)) or values.size != space.size_:
            error_msg = f'The region score of sampling {self.label!r} reads a label per site of a lattice or a network.'
            raise ValidationError(error_msg)
        labels = np.rint(values).astype(np.intp).ravel()
        if labels.size and labels.min() < 0:
            error_msg = f'The region score of sampling {self.label!r} reads labels from 0, but got {labels.min()}.'
            raise ValidationError(error_msg)
        intensity = float(SR(self._get_intensity(SimpleNamespace(**parameters))))
        if self.cooling is not None:
            intensity /= float(SR(self.cooling(time)))
        neighbors, counts, weights, sizes = self._get_contacts(space, self.neighborhood)
        ids = np.arange(int(labels.max(initial=-1)) + 1)
        contact = score.contact(ids[:, None], ids[None, :], **self._get_keywords(score.contact, parameters))
        contact = np.ascontiguousarray(np.broadcast_to(np.asarray(contact, dtype=float), (ids.size, ids.size)))
        constraints = np.stack(
            [
                *self._get_constraint(score.volume, ids, parameters),
                *self._get_constraint(score.surface, ids, parameters),
            ],
        )
        foreign = (weights * (labels[neighbors] != labels[:, None])).sum(axis=1)
        totals = np.stack(
            [
                np.bincount(labels, weights=sizes, minlength=ids.size),
                np.bincount(labels, weights=foreign, minlength=ids.size),
            ],
        )
        attempts = score.hits * labels.size
        _copy_labels(
            labels,
            (neighbors, counts, weights, sizes),
            (contact, constraints, totals),
            (rng.integers(labels.size, size=attempts), rng.random(attempts), rng.random(attempts)),
            (intensity, self.rule == 'metropolis'),
        )
        return labels.reshape(values.shape).astype(values.dtype)

    def _get_keywords(self, function: Callable[..., object], parameters: Mapping[str, object]) -> dict[str, object]:
        """Return the parameters that a function names."""
        try:
            names = inspect.signature(function).parameters
        except (TypeError, ValueError):
            return {}
        return {name: value for name, value in parameters.items() if name in names}

    def _check(self) -> None:
        """Raise an error for an unknown move, rule, or schedule, or an objective that is not callable.

        Raises:
            ValidationError:
                The move, the rule, or the schedule is unknown, or the objective or the cooling is not callable.
        """
        super()._check()
        if self.cooling is not None and not callable(self.cooling):
            error_msg = f'The cooling of sampling {self.label!r} must be a function of the time symbol.'
            raise ValidationError(error_msg)
        self._check_variant('move', self.move, MOVES, functions=True)
        objective = getattr(self, self._get_parameter('objective'))
        if self.move == 'copy' and not isinstance(objective, RegionScoreSpec):
            error_msg = f"The move 'copy' of sampling {self.label!r} copies labels, which a RegionScoreSpec scores."
            raise ValidationError(error_msg)
        if isinstance(objective, RowScoreSpec):
            self._check_row_score(objective)
        elif isinstance(objective, HolonomyScoreSpec):
            self._check_holonomy_score(objective)
        elif isinstance(objective, RegionScoreSpec):
            self._check_region_score(objective)
        elif not callable(objective):
            error_msg = f'The objective of sampling {self.label!r} must be a function of the namespace.'
            raise ValidationError(error_msg)
        SamplerSpec(self.rule, schedule=self.schedule)._validated()

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the sample law of the state.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target` and `namespace.intensity`.

        Returns:
            laws:
                The sample law.
        """
        target = getattr(self._get_member(namespace, 'target'), self.state)
        declared = getattr(self, self._get_parameter('objective'))
        if isinstance(declared, RowScoreSpec):
            return [DriveLaw(target, self._accept_rows, label=self.label, score=self._score_rows)]
        if isinstance(declared, HolonomyScoreSpec):
            return [DriveLaw(target, self._sample_holonomies, label=self.label)]
        if isinstance(declared, RegionScoreSpec):
            return [DriveLaw(target, self._sample_regions, label=self.label)]
        over = VonNeumannNeighborhood() if self.neighborhood is None else self.neighborhood
        objective = ObjectiveSpec(declared(namespace), over)
        intensity = self._get_intensity(namespace)
        if self.cooling is not None:
            intensity = SR(intensity) / SR(self.cooling(namespace.t))
        sampler = SamplerSpec(self.rule, intensity, self.schedule)
        return [SampleLaw(objective, self._get_move(target), sampler, label=self.label)]
