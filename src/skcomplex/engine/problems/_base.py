"""Define the bases of what a system compiles to: values, problems, kernels, searches, printers, records, structures."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING, Self

import numpy as np
import numpy.typing as npt
import shapely
from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from ...base import BaseConfig
from ...core.spaces import BaseSpace
from ...core.system import System
from ...errors import ValidationError
from ...specs import BaseSpec, ObservableSpec
from ...types import AnyArray, Array, Getter

if TYPE_CHECKING:
    from ._compiled_agents import CompiledAgents
    from ._compiled_element import CompiledElement
    from ._compiled_interaction import CompiledInteraction
    from ._form import Form
    from ._layout import Layout
    from ._markov_kernel import MarkovKernel
    from ._partition import Partition
    from ._renewal_kernel import RenewalKernel
    from ._topology_kernel import TopologyKernel
    from ._transfer_network import TransferNetwork
    from ._variable import Variable


class BaseKernel(BaseConfig):
    """Base of the kernels: laws lowered to numbers, the parts of a problem that a solver runs at a state.

    A kernel stores in its constructor what the compiler lowered, such as compiled functions and the indices they read.
    `check()` validates them, `prepare(problem)` sets what a run needs, such as the first renewal times, and
    `run(state)` acts at a `State`, the time, the state vector, the parameters, and the random number generator, and
    returns what the kernel computes, such as the next state or a matrix.

    A developer adds a law kind by subclassing it, together with the pass that builds the kernel and a solver that runs
    it.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.problems import BaseKernel, State
        >>> class DecayKernel(BaseKernel):
        ...     def __init__(self, rate=1.0):
        ...         self.rate = rate
        ...
        ...     def _run(self, inputs):
        ...         return -self.rate * inputs.x
        >>> DecayKernel(0.5).check().prepare().run(State(0.0, np.array([2.0]), np.zeros(0))).tolist()
        [-1.0]
    """


class BaseSearch(BaseConfig):
    """Base of the neighbour searches: the neighbours a kernel reads at a state.

    A search stores the entries and the neighbourhood it reads in its constructor, and `run(state)` returns the
    neighbours at a `State`: the row starts and the neighbours of every member, or the members of every pair.
    """


class BasePrinter(BaseConfig):
    """Base of the printers: they print Sage expressions as code of one target.

    A printer stores its target in its constructor, `prepare(symbols)` takes the symbols the code reads, in order, and
    `run(expressions)` returns the code of the expressions, or a function that evaluates them.
    """


@dataclass(frozen=True)
class BaseRecord(BaseSpec):
    """Base of the records: something that happened at one time of a run, such as an event or a relaxation.

    Every record has the readers `label`, the name of what happened, and `time`, the time it happened at.
    """


@dataclass(frozen=True)
class BaseStructure(BaseSpec):
    """Base of the structures of a rewriting element at one generation: a word, a patch of tiles, or a graph.

    The readers `parts`, `segments`, `polygons`, `geometry`, and `raster` describe the structure; subclasses give the
    parts, the segments, and the polygons, and the measures follow from them with Shapely. Every subclass has the
    field `generation`, and its arrays are read-only, built with `_frozen`, so later generations cannot change a
    recorded structure.
    """

    @property
    def parts(self) -> int:
        """Number of parts, symbols, tiles, or nodes."""
        raise NotImplementedError

    @property
    def has_geometry(self) -> bool:
        """Whether the structure has a geometry, a patch or a word with a word geometry."""
        return False

    def segments(self) -> AnyArray:
        """Return the line segments of the geometry.

        Returns:
            segments:
                The array of shape `(k, 2, d)`, empty without segments.
        """
        return np.zeros((0, 2, 2))

    def polygons(self) -> AnyArray:
        """Return the polygons of the geometry, one coordinate array of shape `(t, k, 2)` per vertex count `k`.

        Returns:
            polygons:
                An object array of coordinate arrays, empty without polygons.
        """
        return np.zeros(0, dtype=object)

    def geometry(self) -> AnyArray:
        """Return the segments and polygons as an array of Shapely geometries.

        Returns:
            geometry:
                The line strings of the segments followed by the polygons.
        """
        return self._geometry

    @functools.cached_property
    def _geometry(self) -> AnyArray:
        """Line strings of the segments followed by the polygons, built once."""
        segments = self.segments()
        lines = shapely.linestrings(segments[..., :2]) if len(segments) else np.zeros(0, dtype=object)
        shapes = [shapely.polygons(coordinates) for coordinates in self.polygons()]
        return np.concatenate([lines, *shapes]) if shapes else lines

    def length(self) -> float:
        """Return the total length of the segments and the perimeters of the polygons.

        Returns:
            length:
                The total length.
        """
        segments = self.segments()
        if segments.shape[-1] == 3:  # noqa: PLR2004
            return float(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=-1).sum())
        return float(shapely.length(self.geometry()).sum())

    def area(self) -> float:
        """Return the total area of the polygons.

        Returns:
            area:
                The sum of the areas of the polygons, 0 without polygons.
        """
        return float(shapely.area(self.geometry()).sum())

    def bounds(self) -> tuple[float, float, float, float]:
        """Return the bounding box of the geometry in the plane.

        Returns:
            bounds:
                `(xmin, ymin, xmax, ymax)`, or zeros without geometry.
        """
        geometry = self.geometry()
        if not len(geometry):
            return (0.0, 0.0, 0.0, 0.0)
        box = shapely.bounds(geometry)
        return (
            float(box[:, 0].min()),
            float(box[:, 1].min()),
            float(box[:, 2].max()),
            float(box[:, 3].max()),
        )

    def raster(self, resolution: int, bounds: Sequence[float] | None = None) -> npt.NDArray[np.bool_]:
        """Return a square boolean grid of the cells the geometry covers, with the first row at the top.

        Segments mark the cells of points spaced at most half a cell apart along them, and polygons mark the cells
        whose centres they contain, so the grid suits `ScalingAnalysis('box-counting')`.

        Args:
            resolution:
                The number of cells per side.

            bounds:
                The box `(xmin, ymin, xmax, ymax)` to rasterise, or `None` for the bounding box of the geometry. The
                grid covers a square with the longer side of the box.

        Returns:
            grid:
                The boolean grid of shape `(resolution, resolution)`.

        Raises:
            ValidationError:
                The resolution is not a positive integer.
        """
        if not isinstance(resolution, int | np.integer) or resolution < 1:
            error_msg = f'resolution = {resolution!r} must be a positive integer.'
            raise ValidationError(error_msg)
        xmin, ymin, xmax, ymax = self.bounds() if bounds is None else bounds
        side = max(xmax - xmin, ymax - ymin) or 1.0
        cell = side / resolution
        grid = np.zeros((resolution, resolution), dtype=bool)
        geometry = self.geometry()
        if not len(geometry):
            return grid
        lines = geometry[shapely.get_type_id(geometry) == shapely.GeometryType.LINESTRING]
        if len(lines):
            points = shapely.get_coordinates(shapely.segmentize(lines, cell / 2))
            self._mark(grid, points, xmin, ymax, cell)
        shapes = geometry[shapely.get_type_id(geometry) == shapely.GeometryType.POLYGON]
        if len(shapes):
            centres = (np.arange(resolution) + 0.5) * cell
            x, y = np.meshgrid(xmin + centres, ymax - centres)
            tree = shapely.STRtree(shapes)
            found = tree.query(shapely.points(x.ravel(), y.ravel()), predicate='intersects')[0]
            grid.ravel()[np.unique(found)] = True
        return grid

    @staticmethod
    def _mark(grid: npt.NDArray[np.bool_], points: AnyArray, xmin: float, ymax: float, cell: float) -> None:
        """Mark the cells of a grid that hold points."""
        resolution = grid.shape[0]
        columns = np.clip(((points[:, 0] - xmin) / cell).astype(int), 0, resolution - 1)
        rows = np.clip(((ymax - points[:, 1]) / cell).astype(int), 0, resolution - 1)
        grid[rows, columns] = True

    @staticmethod
    def _frozen(array: npt.ArrayLike, dtype: type) -> AnyArray:
        """Return a read-only copy of an array, so later generations cannot change a recorded structure."""
        result: AnyArray = np.array(array, dtype=dtype)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class BaseProblem(BaseSpec):
    """Base of the problems a system compiles to: what every solver, observer, and visual reads of one.

    A problem is a value. Its fields hold what the runtime reads, with trivial defaults for what a form does not have,
    and its readers answer the runtime's questions, trivially by default. A formalism's problem adds the fields its
    own solver reads.

    Args:
        form:
            The mathematical form.

        states:
            The layout of the state vector.

        parameters:
            The layout of the parameter vector.

        initial_state:
            The initial state vector.

        initial_parameters:
            The initial parameter vector.

        time:
            The time symbol.

        symbols:
            A mapping from dotted name to the symbol of a scalar or the column of symbols of a vector. The symbols
            of a group stand for one member.

        elements:
            The compiled elements and groups.

        interactions:
            The compiled interactions in tree order.

        system:
            The system the problem was compiled from, or `None`.

        agents:
            The agent elements and groups of an agent-based problem.

        agent_space:
            The space the agents of an agent-based problem live in, or `None`.

        partitions:
            The partitions of a co-simulation problem, in system order.

        companions:
            A mapping from the name of every companion state to its kind, `'derivative'` or `'lag'`, the name of the
            state it belongs to, and the order of the derivative or the number of steps back.

        renewals:
            The renewal laws, whose members renew on their own clocks.

        matrices:
            The transition and emission matrices of the Markov laws, checked when their parameters change.

        transfer_network:
            The quantities, flows, objectives, and feedback terms of the system, or `None` without them.

        topology:
            The topology requests of the laws and the networks they change, or `None` for a problem without them.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.problems import Form
        >>> from skcomplex.engine._utils import symbol
        >>> from skcomplex.engine.problems import Layout
        >>> from skcomplex.engine.problems import BaseProblem
        >>> problem = BaseProblem(Form.ODE, Layout(()), Layout(()), np.zeros(0), np.zeros(0), symbol('t'))
        >>> problem.stochastic, problem.adaptive_parameters, problem.lattice_shape('a.n')
        (False, {}, None)
    """

    form: 'Form'
    states: 'Layout'
    parameters: 'Layout'
    initial_state: Array
    initial_parameters: Array
    time: Expression
    _: KW_ONLY
    symbols: Mapping[str, Expression | FreeModuleElement] = field(default_factory=dict, repr=False)
    elements: tuple['CompiledElement', ...] = ()
    interactions: tuple['CompiledInteraction', ...] = ()
    system: System | None = field(default=None, repr=False)
    agents: tuple['CompiledAgents', ...] = ()
    agent_space: BaseSpace | None = None
    partitions: tuple['Partition', ...] = ()
    companions: Mapping[str, tuple[str, str, int]] = field(default_factory=dict)
    renewals: tuple['RenewalKernel', ...] = field(default=(), repr=False)
    matrices: tuple['MarkovKernel', ...] = field(default=(), repr=False)
    transfer_network: 'TransferNetwork | None' = field(default=None, repr=False)
    topology: 'TopologyKernel | None' = field(default=None, repr=False)

    @property
    def stochastic(self) -> bool:
        """Whether the problem draws random values, so its solver takes the random number generator."""
        return False

    @property
    def observables(self) -> list[tuple[str, ObservableSpec]]:
        """The observables of the system, each with the path of the system that declares it, empty for the root."""
        found: list[tuple[str, ObservableSpec]] = []
        pending = [('', self.system)] if self.system is not None else []
        while pending:
            scope, system = pending.pop()
            found += [(scope, item) for item in getattr(system, 'observables', ())]
            pending += [
                (f'{scope}/{child.label}' if scope else child.label, child) for child in reversed(system.systems)
            ]
        return found

    @property
    def adaptive_parameters(self) -> dict[str, 'Variable']:
        """Declarations of the adaptive parameters, states that a schedule may set, by dotted state name."""
        return {}

    def lattice_shape(self, name: str) -> tuple[int, ...] | None:
        """Return the lattice shape of a state that lives on the sites of a lattice, or `None` for other states.

        Args:
            name:
                The dotted name of the state.

        Returns:
            shape:
                The number of sites along every dimension, or `None`; no state is on a lattice by default.
        """
        del name
        return None

    def observable_getters(self, observables: Iterable[tuple[str, ObservableSpec]]) -> dict[str, Getter] | None:
        """Return the getters of observables by path when the problem evaluates them itself, or `None` otherwise.

        Args:
            observables:
                The path of the declaring system and the observable, for every observable of the system tree.

        Returns:
            getters:
                The getters by observable path, or `None` for the shared evaluator, which reads the fields of the
                problem.
        """
        del observables
        return None

    @staticmethod
    def _check_initial_names(unknown: set[str], names: Sequence[str]) -> None:
        """Raise an error for initial values of states the problem does not have."""
        if unknown:
            error_msg = (
                f'The initial state names {", ".join(sorted(unknown))}, which are not states of the system. Use the '
                f'dotted names of states, such as {", ".join(list(names)[:3])}.'
            )
            raise ValidationError(error_msg)

    def with_initial_state(self, initial_state: Mapping[str, object]) -> Self:
        """Return a problem whose initial state takes given values of some states.

        Args:
            initial_state:
                The initial values by dotted state name.

        Returns:
            problem:
                The problem with the new initial state, or the same problem when no value is given.

        Raises:
            ValidationError:
                A name is not a state of the problem, or a value has the wrong shape.
        """
        if not initial_state:
            return self
        self._check_initial_names(set(initial_state) - set(self.states.names), self.states.names)
        state = self.initial_state.copy()
        for name, value in initial_state.items():
            entry = self.states.entry(name)
            try:
                state[entry.indices] = entry.encode(value)
            except ValueError as error:
                error_msg = f'The initial value of {name} does not fit its shape {entry.shape}: {error}'
                raise ValidationError(error_msg) from error
        return replace(self, initial_state=state)
