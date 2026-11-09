"""Lattice Yang-Mills theory.

SU(2) and SU(3) link matrices are sampled by the Wilson action, pure lattice QCD for SU(3).
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Annotated

import numpy as np
import numpy.typing as npt

from ....core.components.elements import Element, GroupElement
from ....core.components.spaces import CellComplexSpace, DiscreteTime, LatticeSpace
from ....core.components.systems import BaseSystem, RunSettings, System
from ....core.engine.runtime import Trajectory
from ....core.errors import AnalysisError, ValidationError
from ....core.specs import MatrixGroupSpec, ObservableSpec, StateSpec, VisualSpec
from ....core.types import Array, Indices
from ..common.fields import WilsonSamplingInteraction

COLOR = MatrixGroupSpec('SU', 3)


class ColorLinkElement(Element):
    """Link of a lattice with an SU(3) matrix `U`.

    The matrix is the parallel transport of color along the link.

    Args:
        label:
            The label of the link or the group on the links.

        visual:
            How the link is drawn, or `None` for `default_visual_`.

        U:
            The real components of the matrix of the link, the identity by default.

    Attributes:
        default_visual_:
            The visual of the link when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.waves import ColorLinkElement
        >>> len(ColorLinkElement('link').U)
        18
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        U: Annotated[object, StateSpec(group=COLOR)] = tuple(COLOR.identity),
    ) -> None:
        super().__init__(label, visual=visual)
        self.U = U


ISOSPIN = MatrixGroupSpec('SU', 2)


class IsospinLinkElement(Element):
    """Link of a lattice with an SU(2) matrix `U`.

    The matrix is the parallel transport of isospin along the link.

    Args:
        label:
            The label of the link or the group on the links.

        visual:
            How the link is drawn, or `None` for `default_visual_`.

        U:
            The real components of the matrix of the link, the identity by default.

    Attributes:
        default_visual_:
            The visual of the link when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.waves import IsospinLinkElement
        >>> len(IsospinLinkElement('link').U)
        8
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        U: Annotated[object, StateSpec(group=ISOSPIN)] = tuple(ISOSPIN.identity),
    ) -> None:
        super().__init__(label, visual=visual)
        self.U = U


KINDS = ('wilson', 'polyakov', 'charge', 'gauss')
MATRIX_KINDS = ('wilson', 'polyakov')
CHARGE_DIMENSIONS = 2
LOOP_AXES = 2


@dataclass(frozen=True)
class GaugeObservablesResult:
    """Gauge observable of a link state at every frame.

    Args:
        values:
            The link states, with one row per frame.

        value:
            The observable of every frame.
    """

    values: Array
    value: Array


def gauge_observables(
    source: object,
    *,
    kind: str,
    space: CellComplexSpace,
    variables: str | None = None,
    extents: tuple[int, int] = (1, 1),
    charges: Sequence[float] | None = None,
    group: MatrixGroupSpec | None = None,
) -> GaugeObservablesResult:
    """Return a gauge observable of a link state on a periodic lattice's cell complex.

    There is one value per frame.

    The kind `'wilson'` is the mean cosine of the oriented sum of the link angles around the rectangles of given extents
    in every plane. The kind `'polyakov'` is the mean cosine of the sum of the link angles along the last axis. The kind
    `'charge'` is the sum over the plaquettes of a two-dimensional lattice of the oriented sums wrapped into one period
    around 0, over `2 pi`. The kind `'gauss'` is the largest absolute value of the codifferential of an electric link
    state minus a site charge.

    An array of one dimension holds one snapshot, and an array of two dimensions one frame per row.

    With a matrix group, the link states are the real components of the link matrices, and a loop is the path-ordered
    product of the link matrices around it. The kind `'wilson'` is then the mean of `Re tr W / N` over the rectangles of
    every plane, and the kind `'polyakov'` the modulus of the mean of `tr P / N` over the loops `P` that wind once along
    the last axis, the order parameter of deconfinement. An array of two dimensions then holds one snapshot, with the
    components of every link on the last axis, and an array of three dimensions one frame per row.

    Args:
        source:
            A trajectory, or an array of link states.

        kind:
            `'wilson'`, `'polyakov'`, `'charge'`, or `'gauss'`.

        space:
            The cell complex of the links.

        variables:
            The dotted name of the state of a trajectory, or `None` for a source that is an array.

        extents:
            The sides of the rectangles of the kind `'wilson'`.

        charges:
            The charge of every site for the kind `'gauss'`, or `None` for none.

        group:
            The matrix group of the links, or `None` for link angles.

    Returns:
        result:
            The link states and the observable of every frame.

    Raises:
        AnalysisError:
            The kind is unknown, the space is not the cell complex of a periodic lattice, the charge is asked of a
            lattice that does not have two dimensions, or a kind other than a loop is asked of matrix links.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace
        >>> from skcomplex.systems.physics.waves import gauge_observables
        >>> space = CellComplexSpace(LatticeSpace((4, 4))).prepare()
        >>> gauge_observables(np.zeros(32), kind='wilson', space=space).value.tolist()
        [1.0]
        >>> from skcomplex.core.specs import MatrixGroupSpec
        >>> group = MatrixGroupSpec('SU', 2)
        >>> identity = np.tile(group.identity, (32, 1))
        >>> gauge_observables(identity, kind='polyakov', space=space, group=group).value.tolist()
        [1.0]
    """
    if kind not in KINDS:
        error_msg = f'gauge_observables has kind={kind!r}. Use one of {", ".join(KINDS)}.'
        raise AnalysisError(error_msg)
    place = getattr(space, 'space', None)
    if not isinstance(place, LatticeSpace) or place.boundary != 'periodic':
        error_msg = f'gauge_observables reads the cell complex of a periodic lattice, not {space!r}.'
        raise AnalysisError(error_msg)
    if group is not None and kind not in MATRIX_KINDS:
        error_msg = f'The kind {kind!r} reads link angles. Matrix links take the kind wilson or polyakov.'
        raise AnalysisError(error_msg)
    if kind == 'charge' and len(np.ravel(place.shape)) != CHARGE_DIMENSIONS:
        error_msg = f'The topological charge is defined on a two-dimensional lattice, not on {place!r}.'
        raise AnalysisError(error_msg)
    space.prepare()
    if isinstance(source, Trajectory):
        values = np.asarray(source.values(str(variables)), dtype=float)
    else:
        values = np.asarray(source, dtype=float)
        frame = 1 if group is None else 2
        values = values[None] if values.ndim == frame else values
    if group is not None:
        if kind == 'wilson':
            return GaugeObservablesResult(values, _matrix_wilson(space, values, group, extents))
        return GaugeObservablesResult(values, _matrix_polyakov(space, values, group))
    return GaugeObservablesResult(values, _angle_observable(space, values, kind, extents, charges))


def _angle_observable(
    space: CellComplexSpace,
    values: Array,
    kind: str,
    extents: tuple[int, int],
    charges: Sequence[float] | None,
) -> Array:
    """Return the observable of a kind of link angles at every frame."""
    if kind == 'wilson':
        return _wilson(space, values, extents)
    if kind == 'polyakov':
        return _polyakov(space, values)
    if kind == 'charge':
        return np.angle(np.exp(1j * (space.incidences_[1] @ values.T).T)).sum(axis=1) / (2 * np.pi)
    divergence = -(space.incidences_[0].T @ values.T).T / space.stars_[0]
    sites = np.zeros(divergence.shape[1]) if charges is None else np.asarray(charges, dtype=float)
    return np.abs(divergence - sites).max(axis=1)


def _rectangles_of(space: CellComplexSpace, extents: tuple[int, int], first: int, second: int, sums: Array) -> Array:
    """Return the oriented sums around the rectangles of a plane.

    The frame comes first and the sites next.
    """
    shape = space.space.shape_
    faces = np.flatnonzero(np.all(space.face_planes_ == (first, second), axis=1))
    corners = np.asarray([space.hyperedges_[face][0] for face in faces], dtype=np.intp)
    flat = np.zeros((sums.shape[0], int(np.prod(shape))))
    flat[:, corners] = sums[:, faces]
    grid = flat.reshape(sums.shape[0], *shape)
    loops = np.zeros_like(grid)
    for steps in itertools.product(range(extents[0]), range(extents[1])):
        loops += np.roll(grid, (-steps[0], -steps[1]), axis=(first + 1, second + 1))
    return loops


def _get_paths(space: CellComplexSpace, moves: Sequence[tuple[int, int]]) -> tuple[Indices, Array]:
    """Return the links and signs of the path of steps along axes from every site.

    A step is forward 1 or backward -1.
    """
    place = space.space
    sites = np.arange(place.size_).reshape(place.shape_)
    forward = np.stack([np.roll(sites, -1, axis=axis).ravel() for axis in range(place.n_dim_)])
    backward = np.stack([np.roll(sites, 1, axis=axis).ravel() for axis in range(place.n_dim_)])
    table = np.empty((place.n_dim_, place.size_), dtype=np.intp)
    table[space.link_axes_, space.edges_[None][0]] = np.arange(space.link_axes_.size)
    current = np.arange(place.size_)
    edges, signs = [], []
    for axis, direction in moves:
        if direction < 0:
            current = backward[axis, current]
        edges.append(table[axis, current])
        signs.append(np.full(place.size_, float(direction)))
        if direction > 0:
            current = forward[axis, current]
    return np.stack(edges, axis=1), np.stack(signs, axis=1)


def _get_traces(values: Array, group: MatrixGroupSpec, edges: Indices, signs: Array) -> Array:
    """Return the trace over `N` of the path-ordered product along every path.

    There is one row per frame.
    """
    return np.stack(
        [
            np.trace(group.holonomy_of(group.matrices_of(frame), edges, signs), axis1=-2, axis2=-1) / group.degree
            for frame in values
        ],
    )


def _matrix_wilson(space: CellComplexSpace, values: Array, group: MatrixGroupSpec, extents: tuple[int, int]) -> Array:
    """Return the mean of `Re tr W / N` over the rectangles of every plane."""
    rows, columns = extents
    planes = np.unique(space.face_planes_, axis=0).reshape(-1, LOOP_AXES)
    loops = []
    for first, second in planes.tolist():
        moves = [(first, 1)] * rows + [(second, 1)] * columns + [(first, -1)] * rows + [(second, -1)] * columns
        loops.append(_get_traces(values, group, *_get_paths(space, moves)).real)
    return np.concatenate(loops, axis=1).mean(axis=1)


def _matrix_polyakov(space: CellComplexSpace, values: Array, group: MatrixGroupSpec) -> Array:
    """Return the modulus of the mean of `tr P / N` over the loops along the last axis.

    The loops wind along the last axis.
    """
    place = space.space
    last = place.n_dim_ - 1
    traces = _get_traces(values, group, *_get_paths(space, [(last, 1)] * place.shape_[last]))
    return np.abs(traces.mean(axis=1))


def _wilson(space: CellComplexSpace, values: Array, extents: tuple[int, int]) -> Array:
    """Return the mean cosine of the rectangles of every plane."""
    sums = (space.incidences_[1] @ values.T).T
    planes = np.unique(space.face_planes_, axis=0).reshape(-1, LOOP_AXES)
    loops = [_rectangles_of(space, extents, int(first), int(second), sums) for first, second in planes]
    axes = tuple(range(1, space.space.n_dim_ + 1))
    return np.mean([np.cos(loop).mean(axis=axes) for loop in loops], axis=0)


def _polyakov(space: CellComplexSpace, values: Array) -> Array:
    """Return the mean cosine of the sums of the link angles along the last axis."""
    place = space.space
    last = place.n_dim_ - 1
    links = np.flatnonzero(space.link_axes_ == last)
    flat = np.zeros((values.shape[0], place.size_))
    flat[:, space.edges_[None][0][links]] = values[:, links]
    grid = flat.reshape(values.shape[0], *place.shape_)
    return np.cos(grid.sum(axis=last + 1)).mean(axis=tuple(range(1, place.n_dim_)))


LINKS = {2: (IsospinLinkElement, ISOSPIN), 3: (ColorLinkElement, COLOR)}


def _mean_plaquette(space: CellComplexSpace, group: MatrixGroupSpec, t: float, values: Mapping[str, object]) -> float:
    """Return the mean of `Re tr U_p / N` over the plaquettes."""
    del t
    matrices = group.matrices_of(np.reshape(np.asarray(values['links.U'], dtype=float), (-1, group.size)))
    plaquettes = group.holonomy_of(matrices, space.boundaries_, space.boundary_signs_)
    return float(np.trace(plaquettes, axis1=-2, axis2=-1).real.mean() / group.degree)


def _polyakov_loop(space: CellComplexSpace, group: MatrixGroupSpec, t: float, values: Mapping[str, object]) -> float:
    """Return the modulus of the mean of `tr P / N` over the loops along the last axis.

    The loops wind along the last axis.
    """
    del t
    links = np.reshape(np.asarray(values['links.U'], dtype=float), (-1, group.size))
    return float(gauge_observables(links, kind='polyakov', space=space, group=group).value[0])


class LatticeYangMillsSystem(BaseSystem):
    """SU(2) or SU(3) matrices on the links of a periodic lattice.

    The matrices are sampled by the Wilson action.

    The link matrices sample the Wilson action `-(beta / N) sum_p Re tr U_p` under discrete time, where `U_p` is the
    ordered product of the link matrices around the plaquette `p`, with the inverse of a link traversed against its
    orientation. Every sweep proposes `hits` Metropolis moves per link, each a product with a random matrix of the group
    near the identity, and projects the links back onto the group. SU(2) is the isospin connection of the weak force,
    and SU(3) the color connection of the strong force, so that SU(3) is pure lattice QCD. The last axis has `extent`
    sites, the inverse temperature of the lattice, along which the Polyakov loop winds. In two dimensions the plaquettes
    are independent, and the SU(2) mean plaquette at large size is `I_2(beta) / I_1(beta)`.

    Args:
        size:
            The number of sites along every axis but the last.

        dimension:
            The number of axes of the lattice.

        extent:
            The number of sites along the last axis, or `None` for `size`.

        colors:
            The degree `N` of the group SU(N), 2 or 3.

        beta:
            The inverse coupling of the Wilson action, `2 N / g^2`.

        width:
            The spread around the identity of the matrices that move a link.

        hits:
            The number of moves proposed per link and sweep.

        links:
            The initial link matrices, one `N` by `N` complex matrix per link, or `None` for matrices drawn from the
            Haar measure of the group.

        seed:
            The seed of the initial link matrices.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The number of colors is not 2 or 3, or the initial links do not hold one matrix of the group per link.

    Examples:
        >>> from skcomplex.systems.physics.waves import LatticeYangMillsSystem
        >>> [element.label for element in LatticeYangMillsSystem(2, dimension=2).prepare().system_.elements]
        ['links']
        >>> LatticeYangMillsSystem(2, dimension=2, colors=3).prepare().system_.elements[0].element.__name__
        'ColorLinkElement'
    """

    field = 'physics/waves'
    topics = ('gauge', 'lattice', 'monte carlo', 'yang-mills', 'qcd', 'cell complex')
    references = ('Satz 2022', 'Schwichtenberg 2020', 'Wilson 1974', 'Creutz 1980')
    until = 100
    dt = 1

    def __init__(
        self,
        size: int = 6,
        *,
        dimension: int = 4,
        extent: int | None = None,
        colors: int = 2,
        beta: float = 2.3,
        width: float = 0.5,
        hits: int = 4,
        links: npt.ArrayLike | None = None,
        seed: int = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.dimension = dimension
        self.extent = extent
        self.colors = colors
        self.beta = beta
        self.width = width
        self.hits = hits
        self.links = links
        self.seed = seed

    @staticmethod
    def _yang_mills_observables(system: System) -> list[ObservableSpec]:
        """Return the mean plaquette and the Polyakov loop of the links."""
        links = next(element for element in system.elements if element.label == 'links')
        space = links.space.prepare()
        group = next(group for element, group in LINKS.values() if links.element is element)
        return [
            ObservableSpec('mean_plaquette', function=partial(_mean_plaquette, space, group)),
            ObservableSpec('polyakov_loop', function=partial(_polyakov_loop, space, group)),
        ]

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings.

        The observables of the parts are built from the model parameters.
        """
        settings = super().get_run_settings()
        settings['observables'] = tuple(self._yang_mills_observables(self._build_system()))
        return settings

    def _build_system(self) -> System:
        """Return SU(2) or SU(3) matrices on the links of a periodic lattice.

        The matrices are sampled by the Wilson action.
        """
        if self.colors not in LINKS:
            error_msg = f'The lattice gauge group is SU(2) or SU(3), so colors is 2 or 3, but got {self.colors!r}.'
            raise ValidationError(error_msg)
        element, group = LINKS[self.colors]
        shape = (self.size,) * (self.dimension - 1) + (self.size if self.extent is None else self.extent,)
        space = CellComplexSpace(LatticeSpace(shape)).prepare()
        count = space.incidences_[0].shape[0]
        if self.links is None:
            matrices = group.uniform_of(np.random.default_rng(self.seed), count)
        else:
            matrices = np.asarray(self.links, dtype=complex)
        if matrices.shape != (count, self.colors, self.colors) or not np.allclose(
            group.projection_of(matrices),
            matrices,
        ):
            error_msg = (
                f'The initial links are {matrices.shape[0]} matrices of the shape {matrices.shape[1:]}, but the '
                f'lattice takes {count} matrices of SU({self.colors}), one per link.'
            )
            raise ValidationError(error_msg)
        group_links = GroupElement(
            'links',
            element=element,
            space=space,
            on='edges',
            values={'U': group.components_of(matrices)},
        )
        sampling = WilsonSamplingInteraction(
            'wilson',
            target='links',
            beta=self.beta,
            state='U',
            width=self.width,
            group=group,
            hits=self.hits,
        )
        return System(self.label, elements=[group_links], interactions=[sampling], time=DiscreteTime())
