"""Monte Carlo sampling of angle or matrix link variables.

The sampling uses the local Wilson action of a link.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated, ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, cos, oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import SamplingInteraction
from .....core.components.laws import ArrayUpdateLaw, BaseLaw, SampleLaw
from .....core.components.laws.utils import exterior_derivative, neighbors
from .....core.components.spaces import BaseSpace, CellComplexSpace, CofaceNeighborhood
from .....core.errors import ValidationError
from .....core.specs import BaseSpec, MatrixGroupSpec, ParameterSpec, RoleSpec
from .....core.types import Array, ComplexArray, Indices

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
        >>> from skcomplex.core.specs import MatrixGroupSpec
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


class WilsonSamplingInteraction(SamplingInteraction):
    """Monte Carlo sampling of link variables.

    The sampling uses the local Wilson action of the plaquettes of a link.

    For link angles the action is `-beta sum_p cos(theta_p)`, where `theta_p` is the oriented sum of the link angles
    around the plaquette `p`. For the matrices of a group `SU(N)` it is `-(beta / N) sum_p Re tr U_p`, where `U_p` is
    the path-ordered product of the link matrices around `p`, and the move `'rotate'` multiplies a link by a random
    matrix of the group near the identity of spread `width`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group on the edges of a cell complex.

        beta:
            The inverse coupling of the Wilson action.

        state:
            The name of the state that holds the link angles.

        move:
            `'rotate'` for U(1) or a matrix group, or `'random-value'` for Z_N.

        values:
            The angles a random-value move draws from.

        width:
            The largest change of a rotation.

        rule:
            The acceptance, `'metropolis'`, `'glauber'`, or `'heat-bath'`.

        schedule:
            `'colored'` or `'random-sequential'`.

        group:
            The matrix group of the links, or `None` for link angles.

        hits:
            The number of moves proposed per link and sweep of matrix links.

    Examples:
        >>> from skcomplex.systems.physics.common.fields import WilsonSamplingInteraction
        >>> WilsonSamplingInteraction('wilson', target='links', beta=2.0).beta
        2.0
        >>> from skcomplex.core.specs import MatrixGroupSpec
        >>> color = WilsonSamplingInteraction('wilson', target='links', state='U', group=MatrixGroupSpec('SU', 3))
        >>> color.objective.group.degree
        3
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (SampleLaw, ArrayUpdateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        state: str = 'theta',
        move: str = 'rotate',
        values: Sequence[float] = (),
        width: float = 1.5,
        rule: str = 'metropolis',
        schedule: str = 'colored',
        group: MatrixGroupSpec | None = None,
        hits: int = 1,
    ) -> None:
        super().__init__(
            label,
            target=target,
            state=state,
            objective=self._get_action if group is None else HolonomyScoreSpec(self._get_trace_action, group, hits),
            move=move,
            rule=rule,
            schedule=schedule,
            values=values,
            width=width,
        )
        self.beta = beta
        self.group = group
        self.hits = hits

    def _get_action(self, namespace: SimpleNamespace) -> Expression:
        """Return the terms of the Wilson action that hold the angle of a link."""
        theta = getattr(namespace.target, self.state)
        return -namespace.beta * neighbors(cos(exterior_derivative(theta)), over=CofaceNeighborhood())

    def _get_trace_action(self, holonomies: ComplexArray, beta: float) -> Array:
        """Return the Wilson action `-(beta / N) Re tr U_p` of the plaquette holonomies."""
        degree = holonomies.shape[-1]
        return -beta / degree * np.trace(holonomies, axis1=-2, axis2=-1).real

    def _check_move(self) -> None:
        """Raise an error for an unknown move, with the moves of a matrix group for a holonomy score."""
        if not isinstance(getattr(self, self._get_parameter('objective')), HolonomyScoreSpec):
            super()._check_move()

    def _check_objective(self, objective: object) -> None:
        """Raise an error for a holonomy score the sampling cannot run, or an objective that is not callable."""
        if isinstance(objective, HolonomyScoreSpec):
            self._check_holonomy_score(objective)
        else:
            super()._check_objective(objective)

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

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the heat bath of the matrices with a holonomy score, or the Monte Carlo sweep otherwise.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target`.

        Returns:
            laws:
                The law of the sampling.
        """
        if isinstance(getattr(self, self._get_parameter('objective')), HolonomyScoreSpec):
            target = getattr(self._get_member(namespace, 'target'), self.state)
            return [ArrayUpdateLaw(target, self._sample_holonomies, label=self.label)]
        return super()._get_laws(namespace)
