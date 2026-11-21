"""Monte Carlo sampling of angle or matrix link variables.

The sampling uses the local Wilson action of a link.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, cos, oo
from sage.sets.real_set import RealSet

from .....core.base import BaseSpec
from .....core.components import MatrixGroupSpec
from .....core.components.base import BaseComponent
from .....core.components.interactions import Interaction
from .....core.components.laws import ObjectiveLaw, UpdateLaw
from .....core.components.laws.utils import exterior_derivative, neighbors
from .....core.components.spaces import BaseSpace, CellComplexSpace, CofaceNeighborhood
from .....core.errors import ValidationError
from .....core.types import Array, ComplexArray, Indices, Role

HOLONOMY_MOVES = ('rotate', 'random-value')
HOLONOMY_SCHEDULES = ('colored', 'random-sequential')
COLORING_STRATEGY = 'largest_first'
MATRIX_AXES = 2


@dataclass(frozen=True)
class HolonomyScoreSpec(BaseSpec):
    """Spec of an objective of a matrix group state on the edges of a cell complex.

    The objective is a sum over the holonomies of faces. The objective is the sum over the faces of a class function of
    the holonomy of every face, the path-ordered product of the matrices of the edges around it. A class function, such
    as the real part of the trace, has the same value on every cyclic order of the product, so the local objective of an
    edge sums it over the holonomies of the faces around the edge that start at the edge.

    Args:
        function:
            The function `function(holonomies, **parameters)` of an array of holonomies, matrices on the last two axes,
            and the parameters of the system that it names, which returns one number per holonomy.

        group:
            The matrix group of the state.

        hits:
            The number of moves proposed per edge and sweep.

    Raises:
        ValidationError:
            The function is not callable, the group is invalid, or the number of hits is not positive.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.components import MatrixGroupSpec
        >>> from skcomplex.systems.physics.common.fields import HolonomyScoreSpec
        >>> trace = lambda holonomies: -np.trace(holonomies, axis1=-2, axis2=-1).real
        >>> HolonomyScoreSpec(trace, MatrixGroupSpec('SU', 2), hits=4).hits
        4
    """

    function: Callable[..., object]
    group: MatrixGroupSpec
    hits: int = 1

    def _check(self) -> None:
        """Raise an error for a function not callable, a bad group, or a bad hit count.

        The number of hits must be at least 1.

        Raises:
            ValidationError:
                The function is not callable, the group is invalid, or the number of hits is not positive.
        """
        if not callable(self.function):
            error_msg = f'The function of a holonomy score must be callable, but it is {self.function!r}.'
            raise ValidationError(error_msg)
        if not isinstance(self.group, MatrixGroupSpec):
            error_msg = f'The group of a holonomy score is a MatrixGroupSpec, but it is {self.group!r}.'
            raise ValidationError(error_msg)
        self.group._check()
        if not isinstance(self.hits, int) or self.hits < 1:
            error_msg = f'A holonomy score proposes at least one move per edge, but it has hits={self.hits!r}.'
            raise ValidationError(error_msg)


class WilsonSamplingInteraction(Interaction):
    """Boltzmann weight of link variables under the local Wilson action of the plaquettes of a link.

    For link angles the action is `-beta sum_p cos(theta_p)`, where `theta_p` is the oriented sum of the link angles
    around the plaquette `p`. It is the objective of every link, which the `MetropolisSolver` of the system moves, such
    as `MetropolisSolver(moves={'links.theta': 'rotate'}, schedule='colored', width=1.5)` for U(1) or
    `moves={'links.theta': 'random-value'}` for Z_N. For the matrices of a group `SU(N)` the action is
    `-(beta / N) sum_p Re tr U_p`, where `U_p` is the path-ordered product of the link matrices around `p`, and the
    interaction sweeps the links itself: the move `'rotate'` multiplies a link by a random matrix of the group near the
    identity of spread `width`, and `'random-value'` draws a uniform matrix of the group, `hits` times per link.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group on the edges of a cell complex.

        beta:
            The inverse coupling of the Wilson action.

        state:
            The name of the state that holds the link angles or matrices.

        group:
            The matrix group of the links, or `None` for link angles.

        move:
            The move of matrix links, `'rotate'` or `'random-value'`, or a function of the matrices and the random
            generator.

        width:
            The spread of a rotation of matrix links.

        rule:
            The acceptance of the moves of matrix links, `'metropolis'` or `'glauber'`.

        schedule:
            The order of the moves of matrix links, `'colored'` or `'random-sequential'`.

        hits:
            The number of moves proposed per matrix link and sweep.

    Examples:
        >>> from skcomplex.systems.physics.common.fields import WilsonSamplingInteraction
        >>> WilsonSamplingInteraction('wilson', target='links', beta=2.0).beta
        2.0
        >>> from skcomplex.core.components import MatrixGroupSpec
        >>> color = WilsonSamplingInteraction('wilson', target='links', state='U', group=MatrixGroupSpec('SU', 3))
        >>> color.group.degree
        3
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        beta: Annotated[float, RealSet.closed_open(0, oo)] = 2.0,
        state: str = 'theta',
        group: MatrixGroupSpec | None = None,
        move: str | Callable[..., object] = 'rotate',
        width: float = 1.5,
        rule: str = 'metropolis',
        schedule: str = 'colored',
        hits: int = 1,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.beta = beta
        self.state = state
        self.group = group
        self.move = move
        self.width = width
        self.rule = rule
        self.schedule = schedule
        self.hits = hits

    def _get_score(self) -> HolonomyScoreSpec | None:
        """Return the holonomy score of matrix links, or `None` for link angles."""
        return None if self.group is None else HolonomyScoreSpec(self._get_trace_action, self.group, self.hits)

    def _get_action(self, namespace: SimpleNamespace) -> Expression:
        """Return the terms of the Wilson action that hold the angle of a link."""
        theta = getattr(namespace.target, self.state)
        return -namespace.beta * neighbors(cos(exterior_derivative(theta)), over=CofaceNeighborhood())

    def _get_trace_action(self, holonomies: ComplexArray, beta: float) -> Array:
        """Return the Wilson action `-(beta / N) Re tr U_p` of the plaquette holonomies."""
        degree = holonomies.shape[-1]
        return -beta / degree * np.trace(holonomies, axis1=-2, axis2=-1).real

    @staticmethod
    def _get_keywords(function: Callable[..., object], parameters: Mapping[str, object]) -> dict[str, object]:
        """Return the parameters that a function names."""
        try:
            names = inspect.signature(function).parameters
        except (TypeError, ValueError):
            return {}
        return {name: value for name, value in parameters.items() if name in names}

    def _check_arguments(self) -> None:
        """Raise an error for options matrix links do not take.

        Raises:
            ValidationError:
                The holonomy score of matrix links has a move, a rule, or a schedule they do not take.
        """
        super()._check_arguments()
        score = self._get_score()
        if score is not None:
            self._check_holonomy_score(score)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the action or the sweep of the links.

        Args:
            context:
                The namespace of the links, the time `t`, and the parameters.
        """
        super()._prepare(context)
        links = getattr(self._get_member(context, 'target'), self.state)
        if self._get_score() is None:
            self.laws_: list[BaseComponent] = [ObjectiveLaw(self._get_action(context), links)]
        else:
            self.laws_ = [UpdateLaw(links, self._sample_holonomies, label=self.label, arrays=True)]

    def _check_holonomy_score(self, score: HolonomyScoreSpec) -> None:
        """Raise an error for a holonomy score with options a matrix group does not take.

        The options are a move, a rule, or a schedule.
        """
        score._check()
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
        """Return the tables of the faces around every edge of a cell complex.

        They are read once per complex.

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
        """Return the local objective of every edge.

        It is the score of the holonomies of its faces that start at it.
        """
        score = self._get_score()
        assert score is not None
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
            return group.get_uniform_of(rng, len(matrices))
        return group.get_near_identity_of(rng, self.width, len(matrices)) @ matrices

    def _accept_changes(self, changes: Array, intensity: float, rng: np.random.Generator) -> npt.NDArray[np.bool_]:
        """Return whether the rule accepts every change of an objective at the intensity."""
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
        """Return a matrix group state on the edges of a cell complex after one sweep.

        The sweep makes moves of every edge.
        """
        score = self._get_score()
        if (
            score is None
            or not isinstance(space, CellComplexSpace)
            or values.ndim != MATRIX_AXES
            or values.shape[0] != len(space.cofaces_[1])
        ):
            error_msg = (
                f'The holonomy score of sampling {self.label!r} reads a group on the edges of a CellComplexSpace.'
            )
            raise ValidationError(error_msg)
        intensity = 1.0
        group = score.group
        matrices = group.get_matrices_of(values)
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
            staples = group.get_holonomy_of(matrices, edges[links], signs[links])
            current = self._get_local_scores(matrices[links], staples, orientations[links], real[links], local)
            for _ in range(score.hits):
                proposal = self._propose_matrices(matrices[links], group, random)
                proposed = self._get_local_scores(proposal, staples, orientations[links], real[links], local)
                accepted = self._accept_changes(proposed - current, intensity, random)
                matrices[links[accepted]] = proposal[accepted]
                current = np.where(accepted, proposed, current)
        return group.get_components_of(group.get_projection_of(matrices))
