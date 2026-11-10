"""Run agent-based problems on the Mesa backend, one Mesa step per unit of time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
import numbers
from collections import deque
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import cast, overload

import mesa
import networkx as nx
import numpy as np
import numpy.typing as npt
import shapely
from mesa.discrete_space import CellAgent, Network, OrthogonalMooreGrid, OrthogonalVonNeumannGrid
from mesa.experimental.continuous_space import ContinuousSpace as MesaContinuousSpace
from mesa.experimental.continuous_space import ContinuousSpaceAgent
from scipy import stats
from shapely import Point, Polygon, STRtree

from ...components.elements import BaseElement, GroupElement
from ...components.laws import BaseLaw, BoundaryLaw, MatchLaw, RuleLaw, TransferLaw
from ...components.spaces import (
    BaseSpace,
    ContinuousSpace,
    DelaunayNeighborhood,
    DiscreteTime,
    LatticeSpace,
    NetworkSpace,
)
from ...components.specs import CompiledAgentsSpec
from ...components.systems import CompositeSystem
from ...errors import ConservationError, SimulationError, SpaceError, ValidationError
from ...specs import (
    DEFAULT_PHASE,
    AgentSnapshotSpec,
    AgentStateSpec,
    DrawSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    SupportSpec,
)
from ...types import Array, FormKind, OrderKind, VariableKind
from .._utils import declarations_of, default_of
from ..compile._observables import ObservableEvaluator
from ..problems import ProblemSpec
from ..problems._draw_kernel import DrawKernel
from ..problems._draw_recorder import KEYWORDS
from ..problems._metric_search import MetricSearch
from ..types import MesaSpace
from ._external import ExternalSolver

MAX_SEED = 2**31 - 1
HISTORY_LENGTH = 10_000
CONTINUOUS_SIDES = (2, 3)
MIN_CONTINUOUS_ROOM = 2


class _AgentSystem(mesa.Model):
    """Mesa system whose step runs the agent and system rules of a stepper.

    Mesa's clock thus moves with the run.
    """

    def __init__(self, run_rules: Callable[[], None], rng: int) -> None:
        self._run_rules = run_rules
        super().__init__(rng=rng)

    def step(self) -> None:
        """Run the agent rules in random order and then the system rules."""
        self._run_rules()


class SystemView:
    """System as rules see it: parameters, random numbers, time, agents, and the space.

    Every rule receives the view scoped to the interaction or agent group that owns it, so the rule reads the owner's
    parameters by name and the groups bound to the owner's roles by role, also when its system is nested in another.

    Args:
        stepper:
            The stepper that runs the system.

        owner:
            The path of the interaction or agent group whose rules receive the view, or `''` for no owner.

        roles:
            The paths of the elements and groups the owner's roles bind to, by role.
    """

    def __init__(
        self,
        stepper: 'MesaSolver',
        owner: str = '',
        roles: Mapping[str, str | tuple[str, ...]] | None = None,
    ) -> None:
        self._stepper = stepper
        self._owner = owner
        self._roles = dict(roles or {})

    def _get_paths(self, name: str) -> list[str]:
        """Return the paths of the groups a role of the owner binds to.

        A group given by relative or absolute path is resolved too.
        """
        bound = self._roles.get(name)
        if bound is not None:
            return [bound] if isinstance(bound, str) else list(bound)
        try:
            return [self._stepper._resolve_group(name, self._owner)]
        except SimulationError as error:
            roles, owner = ', '.join(self._roles) or 'none', self._owner or 'the system'
            error_msg = f'No role or agent group named {name!r} for {owner}. Its roles are: {roles}. {error}'
            raise SimulationError(error_msg) from error

    def get_parameter(self, name: str) -> float:
        """Return the current value of a parameter of the owner.

        One component is read as `name[i]`.

        Args:
            name:
                The name of a scalar parameter, such as `threshold`, or of one component, such as `weights[1]`.

        Returns:
            value:
                The value.

        Raises:
            SimulationError:
                The owner has no such parameter, or the parameter has several components and no index.
        """
        stepper = self._stepper
        layout = stepper.problem_.parameters
        try:
            entry, indices = layout.locate(f'{self._owner}.{name}')
        except KeyError as error:
            prefix = f'{self._owner}.'
            names = [item.name[len(prefix) :] for item in layout.entries if item.name.startswith(prefix)]
            error_msg = (
                f'{self._owner or "The system"} has no parameter {name!r}. Its parameters are: '
                f'{", ".join(names) or "none"}.'
            )
            raise SimulationError(error_msg) from error
        if entry.size != 1:
            error_msg = f'{entry.name} has {entry.size} components. Read one of them as {name}[i].'
            raise SimulationError(error_msg)
        return float(stepper.parameters_[indices][0])

    @property
    def parameters(self) -> dict[str, float]:
        """Current values of every parameter by dotted name."""
        stepper = self._stepper
        return {
            name: float(np.ravel(value)[0])
            for name, value in stepper.problem_.parameters.get_values(stepper.parameters_).items()
        }

    @property
    def inputs(self) -> Mapping[str, object]:
        """Values of the states of other partitions at the last synchronisation point.

        They are keyed by dotted name.
        """
        return self._stepper.inputs_

    @property
    def rng(self) -> np.random.Generator:
        """Random number generator of the system, seeded by the simulation's random state."""
        return self._stepper.system_.rng

    def draw(self, name: str, distribution: str, **arguments: npt.ArrayLike) -> float:
        """Return a value drawn by name from a distribution, which runs record and replay.

        The value comes from the generator of the system. A run that records keeps it by the path of the owner and
        the name, such as `meetings.outcome`, with the arguments of the distribution, and a run that replays returns
        the recorded value instead and adds its log-probability.

        Args:
            name:
                The name of the draw, unique among the draws of the owner.

            distribution:
                `'uniform'` with `low` and `high`, `'normal'` with `mean` and `deviation`, `'bernoulli'` with `p`,
                `'categorical'` with `weights`, or the name of a `scipy.stats` distribution with its keywords.

            **arguments:
                The arguments of the distribution by keyword.

        Returns:
            value:
                The drawn value.

        Raises:
            ValidationError:
                The distribution is unknown or an argument is missing.
        """
        if distribution == 'categorical':
            draw, values = DrawSpec('categorical', name), list(np.ravel(arguments.get('weights', [])))
        elif distribution in KEYWORDS:
            missing = [keyword for keyword in KEYWORDS[distribution] if keyword not in arguments]
            if missing:
                error_msg = f'A {distribution} draw takes the arguments {", ".join(KEYWORDS[distribution])}.'
                raise ValidationError(error_msg)
            draw, values = DrawSpec(distribution, name), [arguments[keyword] for keyword in KEYWORDS[distribution]]
        elif isinstance(getattr(stats, distribution, None), stats.rv_continuous | stats.rv_discrete):
            draw, values = DrawSpec('draw', name, distribution, tuple(arguments)), list(arguments.values())
        else:
            error_msg = (
                f'The draw {name!r} takes the distribution {distribution!r}, which is neither uniform, normal, '
                'bernoulli, categorical, nor a scipy.stats distribution.'
            )
            raise ValidationError(error_msg)
        sampled = DrawKernel._sample_draw(draw, values, 1, self.rng)
        recorder = self._stepper.draw_recorder_
        if recorder is not None:
            sampled = recorder.draw(f'{self._owner}.{name}' if self._owner else name, draw, values, sampled)
        return float(sampled[0])

    @property
    def time(self) -> float:
        """Time of Mesa's clock.

        It is `k` during the step that produces frame `k`, plus `i / n` at visit `i` of `n`.
        """
        stepper = self._stepper
        visit, visits = stepper.visit_
        return float(stepper.system_.time) + visit / max(visits, 1)

    @property
    def dimensions(self) -> tuple[float, ...] | None:
        """Extent of the agent space per axis.

        It is cells of a grid or lengths of a continuous space, and `None` otherwise.
        """
        space = self._stepper.space_
        if isinstance(space, MesaContinuousSpace):
            return tuple(float(side) for side in space.size)
        if isinstance(space, OrthogonalMooreGrid | OrthogonalVonNeumannGrid):
            return tuple(int(item) for item in space.dimensions)
        return None

    @property
    def torus(self) -> bool:
        """Whether the agent space wraps around at its edges."""
        return bool(getattr(self._stepper.space_, 'torus', False))

    @overload
    def get_agents(self, name: str) -> list['AgentView']: ...

    @overload
    def get_agents[V: AgentView](self, name: str, view: type[V]) -> list[V]: ...

    def get_agents[V: AgentView](self, name: str, view: type[V] | None = None) -> list[V] | list['AgentView']:
        """Return the views of the agents of a group bound to a role of the owner.

        A group may also be named by path.

        Args:
            name:
                A role of the owner, the path of a group relative to the owner's system, or the path of a group.

            view:
                The view class of the group, which its element names with `_get_view`, or `None` for `AgentView`.

        Returns:
            agents:
                The views of the agents alive in the group, or in every group of a role that binds several.

        Raises:
            SimulationError:
                No role or group has that name, or the views of a group are not of the view class.
        """
        stepper = self._stepper
        views = [
            MesaSolver._get_view_of(agent)
            for path in self._get_paths(name)
            for agent in stepper._get_group_agents(path)
        ]
        wanted = AgentView if view is None else view
        wrong = [item for item in views if not isinstance(item, wanted)]
        if wrong:
            viewed = type(wrong[0]).__mro__[1].__name__
            error_msg = f'The agents of {name} are viewed as {viewed}, not {wanted.__name__}.'
            raise SimulationError(error_msg)
        if view is None:
            return views
        return [item for item in views if isinstance(item, view)]

    def get_groups(self, name: str) -> list[str]:
        """Return the paths of the groups a role of the owner binds.

        A group may also be named by relative or absolute path.

        Args:
            name:
                A role of the owner, the path of a group relative to the owner's system, or the path of a group.

        Returns:
            paths:
                The paths of the groups.

        Raises:
            SimulationError:
                No role or group has that name.
        """
        return self._get_paths(name)

    def count(self, name: str) -> int:
        """Return the number of agents of a group bound to a role of the owner.

        A group may also be named by path.

        Args:
            name:
                A role of the owner, the path of a group relative to the owner's system, or the path of a group.

        Returns:
            count:
                The number of agents alive in the group, or in every group of a role that binds several.

        Raises:
            SimulationError:
                No role or group has that name.
        """
        return sum(len(self._stepper._get_group_agents(path)) for path in self._get_paths(name))

    def get_composition(self, name: str, state: str) -> dict[Hashable, int]:
        """Return the number of agents of a group that hold every value of a state.

        Only the values some agent holds are counted. The stepper keeps the counts as agents are created, removed, and
        change the state, so reading them costs no pass over the agents.

        Args:
            name:
                A role of the owner, the path of a group relative to the owner's system, or the path of a group.

            state:
                The name of a scalar state of the agents of the group.

        Returns:
            counts:
                The number of agents alive in the group, or in every group of a role that binds several, by value.

        Raises:
            SimulationError:
                No role or group has that name, or its agents have no such state.
        """
        counts: dict[Hashable, int] = {}
        for path in self._get_paths(name):
            for value, count in self._stepper._get_composition(path, state).items():
                counts[value] = counts.get(value, 0) + count
        return {value: count for value, count in counts.items() if count}

    def get_occupancy(self, name: str) -> dict[Hashable, int]:
        """Return the number of agents of a group in every cell or node that holds one.

        The counts are keyed by the key of the location.

        Args:
            name:
                A role of the owner, the path of a group relative to the owner's system, or the path of a group.

        Returns:
            counts:
                The number of agents alive in the group, or in every group of a role that binds several, by the
                `location_key` of their cell, node, or position.

        Raises:
            SimulationError:
                No role or group has that name.
        """
        counts: dict[Hashable, int] = {}
        for path in self._get_paths(name):
            for agent in self._stepper._get_group_agents(path):
                key = self._stepper._get_view_of(agent).location_key
                counts[key] = counts.get(key, 0) + 1
        return counts

    def get_regions(self, layer: str) -> int:
        """Return the number of regions of a layer of the continuous space.

        Args:
            layer:
                The name of the layer.

        Returns:
            count:
                The number of regions of the layer.

        Raises:
            SimulationError:
                The space has no such layer.
        """
        return len(self._stepper._get_index_of(layer).geometries)

    def get_region(self, layer: str, index: int) -> Polygon:
        """Return the polygon of a region of a layer of the continuous space.

        A rule can pass the polygon to `move_to`.

        Args:
            layer:
                The name of the layer.

            index:
                The index of the region in its layer.

        Returns:
            polygon:
                The polygon of the region.

        Raises:
            SimulationError:
                The space has no such layer, or the layer has no region of that index.
        """
        polygons = self._stepper._get_index_of(layer).geometries
        if not 0 <= index < len(polygons):
            error_msg = f'The layer {layer!r} has {len(polygons)} regions, and a rule asks for the region {index}.'
            raise SimulationError(error_msg)
        return polygons[index]

    def _get_field_name(self, name: str) -> str:
        """Return the path of a field state.

        The state is named by its path, or as `<role>.<state>` of a role of the owner.
        """
        role, _, state = name.rpartition('.')
        bound = self._roles.get(role)
        return f'{bound}.{state}' if isinstance(bound, str) else name

    def _get_site(self, name: str, cell: object) -> int:
        """Return the member index of a field state's site at a cell or network node."""
        problem = self._stepper.problem_
        if not problem.states.has(name):
            names = [entry.name for entry in problem.states.entries if entry.members is not None]
            error_msg = f'No field state named {name!r}. Available group states are: {", ".join(names)}.'
            raise KeyError(error_msg)
        shape = problem.get_lattice_shape(name)
        if shape is not None:
            return int(np.ravel_multi_index(tuple(int(item) for item in np.atleast_1d(np.asarray(cell))), shape))
        owner = name.rpartition('.')[0]
        element = next(item for item in problem.elements if item.path == owner)
        space = getattr(element.element, 'space', None)
        if space is None or not hasattr(space, 'graph'):
            error_msg = f'{name} is not a state of a field on a lattice or a network.'
            raise KeyError(error_msg)
        return list(space.graph.nodes).index(cell)

    def get_field(self, name: str, cell: object) -> float:
        """Return the value of a field at a cell.

        It is the value at the last synchronisation point.

        Args:
            name:
                The dotted path of the field state, such as `ground/pheromone.level`, or `<role>.<state>` for a field
                bound to a role of the owner.

            cell:
                The lattice coordinates of the cell, which match the coordinates of a grid of the same shape, or a
                network node.

        Returns:
            value:
                The value of the field there.

        Raises:
            KeyError:
                The name is not a state of a field.
        """
        name = self._get_field_name(name)
        site = self._get_site(name, cell)
        values = self._stepper.inputs_.get(name)
        if values is None:
            error_msg = f'The agents read the field {name}, which no other partition provides. Run it in co-simulation.'
            raise KeyError(error_msg)
        return float(np.ravel(np.asarray(values))[site])

    def deposit(self, name: str, cell: object, amount: float) -> None:
        """Add an amount to a field at a cell at the next synchronisation point.

        Args:
            name:
                The dotted path of the field state, such as `ground/pheromone.level`, or `<role>.<state>` for a field
                bound to a role of the owner.

            cell:
                The lattice coordinates of the cell or a network node.

            amount:
                The amount to add.

        Raises:
            KeyError:
                The name is not a state of a field.
        """
        name = self._get_field_name(name)
        site = self._get_site(name, cell)
        entry = self._stepper.problem_.states.get_entry(name)
        self._stepper.deposits_[entry.start + site * entry.columns] += amount

    def get_random_empty_cell(self) -> object:
        """Return the coordinates of a random empty cell of the grid or network.

        Returns:
            location:
                The cell coordinates or the network node.
        """
        return self._stepper._get_cells().select_random_empty_cell().coordinate

    def create(self, group: str, location: object = None, **values: object) -> 'AgentView':
        """Create an agent of a group, at a location or at a random place.

        Args:
            group:
                A role of the owner that binds the group, or the path of the group.

            location:
                The cell coordinates, network node, or position, a polygon for a uniform point inside it, or `None` for
                a random place.

            **values:
                The initial values of states, with the element type's defaults for the others.

        Returns:
            agent:
                The new agent.

        Raises:
            SimulationError:
                The group is not variable, so its size cannot change.
        """
        return MesaSolver._get_view_of(self._stepper._create_agent(self._get_paths(group)[0], location, values))

    @property
    def round(self) -> int:
        """Index of the current round of a repeated phase, `0` outside repeated phases."""
        return self._stepper.round_

    @property
    def storage(self) -> dict[str, object]:
        """Mutable mapping private to the owner that persists across the steps of one run.

        An example is an order book.
        """
        return self._stepper.storage_.setdefault(self._owner, {})

    def _report_name(self, name: str) -> str:
        """Return the reporter name of a name of the owner."""
        return f'{self._owner}.{name}' if self._owner else name

    def report(self, name: str, value: float) -> None:
        """Publish a value of the owner as the reporter `<owner>.<name>`.

        The value is added to the reporter's history. The owner declares the names it reports in its class variable
        `reports`, so every frame records them, `NaN` before the first report.

        Args:
            name:
                The name of the value, such as `price`.

            value:
                The value.

        Raises:
            SimulationError:
                The owner does not declare the name in `reports`.
        """
        full = self._report_name(name)
        if full not in self._stepper.reports_:
            error_msg = (
                f'{self._owner or "The system"} reports {name!r}, which it does not declare. List it in the class '
                'variable `reports` of the interaction or element type.'
            )
            raise SimulationError(error_msg)
        kept = float(value) if isinstance(value, numbers.Real) else value
        self._stepper.reports_[full] = kept
        self._stepper.reported_.setdefault(full, deque(maxlen=HISTORY_LENGTH)).append(kept)

    def get_history(self, name: str, window: int | None = None) -> npt.NDArray[np.float64]:
        """Return the past values of a reporter, or of an observable.

        An observable gives its values at the end of each earlier step, the last one last.

        Args:
            name:
                The name of a value the owner reports, the full name of any reporter, such as `bar.attendance`, or the
                path of an observable of the run, such as `attendance`.

            window:
                The number of last values, or `None` for every recorded value.

        Returns:
            values:
                The values, at most `window` of them.
        """
        stepper = self._stepper
        history = stepper.reported_
        values = history.get(self._report_name(name), history.get(name, stepper.observed_.get(name, ())))
        array = (
            np.asarray(values, dtype=float)
            if all(isinstance(item, numbers.Real) for item in values)
            else np.fromiter(values, dtype=object, count=len(values))
        )
        return array if window is None else array[len(array) - min(window, len(array)) :]

    def get_observable(self, name: str) -> object:
        """Return the current value of an observable of the system.

        An example is an aggregate of a group.

        Args:
            name:
                The path of the observable.

        Returns:
            value:
                Its value for the current agents and parameters.

        Raises:
            SimulationError:
                The system has no such observable.
        """
        return self._stepper._get_observable(name)

    def set_parameter(self, name: str, value: float) -> None:
        """Set a parameter of any group or interaction during the run.

        The trajectory records it like any parameter.

        Args:
            name:
                The dotted name of the parameter, such as `bank.rate`, or the name of a parameter of the owner.

            value:
                The new value.

        Raises:
            SimulationError:
                No parameter has that name.
        """
        layout = self._stepper.problem_.parameters
        full = name if layout.has(name) else f'{self._owner}.{name}'
        if not layout.has(full):
            error_msg = f'No parameter named {name!r} to set. Available names are: {", ".join(layout.names)}.'
            raise SimulationError(error_msg)
        self._stepper._set_parameter(full, value)


class AgentView:
    """Agent as rules see it.

    It shows its states as attributes, its group's parameters, its neighbours, and its location. The stepper gives every
    agent one view of the class of its group, whose states and parameters are properties, so two views of an agent are
    the same object. Reading a state reads the agent, and setting it sets the agent, at the end of the phase in a
    simultaneous phase. Parameters are read only, and an unknown name raises Python's own `AttributeError`.

    Args:
        agent:
            The Mesa agent.

        stepper:
            The stepper that runs the system.
    """

    __slots__ = ('_agent', '_stepper')

    def __init__(self, agent: mesa.Agent, stepper: 'MesaSolver') -> None:
        self._agent = agent
        self._stepper = stepper

    @property
    def id(self) -> int:
        """Unique id of the agent."""
        return int(self._agent.unique_id)

    @property
    def group(self) -> str:
        """Path of the agent's group."""
        return self._stepper._get_spec_of(self._agent).path

    @property
    def group_label(self) -> str:
        """Label of the agent's group, the last part of its path."""
        return self.group.rpartition('/')[2]

    @property
    def location(self) -> object:
        """Grid coordinates, network node, or position of the agent."""
        return MesaSolver._get_location_of(self._agent)

    @property
    def location_key(self) -> Hashable:
        """Location of the agent as a key.

        The key is a tuple of integers for a cell, a tuple of floats for a position, or a node. The key equals the
        coordinates `neighbor_cells` gives for the agent's cell or node, so rules compare the cells around an agent with
        the places of other agents.
        """
        location = self.location
        if isinstance(location, tuple | list | np.ndarray):
            values = np.asarray(location)
            if np.issubdtype(values.dtype, np.integer) or np.all(values == np.round(values)):
                return tuple(int(item) for item in values.ravel())
            return tuple(float(item) for item in values.ravel())
        return location if isinstance(location, Hashable) else repr(location)

    @property
    def alive(self) -> bool:
        """Whether the agent has not been removed."""
        return not getattr(self._agent, 'removed_', False)

    def get_displacement_to(self, other: 'AgentView | object') -> npt.NDArray[np.float64]:
        """Return the vector from the agent to another agent or a location.

        On a torus, it is the shortest one.

        Args:
            other:
                An agent, or grid coordinates or a position.

        Returns:
            displacement:
                One component per axis of the space.

        Raises:
            SpaceError:
                The agents live on a network or in no space.
        """
        system = SystemView(self._stepper)
        dimensions = system.dimensions
        if dimensions is None:
            error_msg = (
                'Displacements need a grid or a continuous space, and the agents live on a network or in no space. '
                'Use `distance_to` for the number of hops on a network.'
            )
            raise SpaceError(error_msg)
        target = other.location if isinstance(other, AgentView) else other
        delta = np.asarray(target, dtype=float) - np.asarray(self.location, dtype=float)
        if not system.torus:
            return delta
        size = np.asarray(dimensions, dtype=float)
        return delta - size * np.round(delta / size)

    def get_distance_to(self, other: 'AgentView | object') -> float:
        """Return the distance to another agent or a location.

        On a network, it is the number of hops.

        Args:
            other:
                An agent, or a network node, grid coordinates, or a position.

        Returns:
            distance:
                The Euclidean length of the shortest displacement, or the length of the shortest path on a network.

        Raises:
            SpaceError:
                The agents live in no space.
        """
        space = self._stepper.problem_.agent_space
        if isinstance(space, NetworkSpace):
            target = other.location if isinstance(other, AgentView) else other
            return float(nx.shortest_path_length(space.graph, self.location, target))
        return float(np.linalg.norm(self.get_displacement_to(other)))

    def get_neighbor_cells(self) -> list[object]:
        """Return the coordinates of the cells or nodes around the agent's cell or node.

        Returns:
            locations:
                The neighbouring cell coordinates or network nodes.
        """
        return [cell.coordinate for cell in self._agent.cell.neighborhood]

    def get_neighbors(
        self,
        radius: float | None = None,
        layer: str | None = None,
        group: str | None = None,
        over: DelaunayNeighborhood | None = None,
    ) -> list['AgentView']:
        """Return the agents around the agent.

        They are the agents in the cells or nodes around it, within a radius, around it in a layer, or joined to it by
        an edge of the Delaunay triangulation of the positions of its group, or of `group`, in a continuous space. The
        triangulation is that of the positions when the group is first read in a step, or after agents of the group are
        created or removed, so the agents of a step read the same one, and in a periodic box it covers the periodic
        images. Agents of a
        variable group placed on the nodes of a network make a set of members per node that grows and shrinks during a
        run, such as the balls of an urn on every node, and an agent on a node reads the sets of the neighbouring nodes
        as the neighbours of one group.

        Args:
            radius:
                The distance within which agents are neighbours in a continuous space.

            layer:
                The layer of a network space whose edges join the agent to its neighbours, or `None` for the graph of
                the network.

            group:
                The path of a group whose agents alone are returned, or `None` for every group.

            over:
                A Delaunay neighbourhood of a continuous space, or `None`.

        Returns:
            agents:
                The neighbouring agents.

        Raises:
            SpaceError:
                A Delaunay neighbourhood is read outside a continuous space.
        """
        stepper = self._stepper
        if over is not None:
            triangulated = stepper._resolve_group(group, self.group) if group is not None else self.group
            return [
                MesaSolver._get_view_of(agent)
                for agent in stepper._get_delaunay_neighbors(self._agent, triangulated, over)
            ]
        if layer is not None:
            graph = getattr(stepper.problem_.agent_space, 'graphs_', {})[layer]
            cells = stepper._get_cells()
            agents = [
                agent
                for node in graph.neighbors(self.location_key)
                for agent in cells[MesaSolver._get_cell_key(node)].agents
            ]
        elif radius is not None:
            found, _ = self._agent.get_neighbors_in_radius(radius)
            agents = [agent for agent in found if agent is not self._agent]
        else:
            agents = list(self._agent.cell.neighborhood.agents)
        path = stepper._resolve_group(group, self.group) if group is not None else None
        return [
            MesaSolver._get_view_of(agent)
            for agent in agents
            if path is None or stepper._get_spec_of(agent).path == path
        ]

    def get_agents_here(self, group: str | None = None) -> list['AgentView']:
        """Return the other agents in the agent's cell or node.

        They are of one group or of every group.

        Args:
            group:
                The path of a group, or `None` for every group.

        Returns:
            agents:
                The other agents at the same place.
        """
        stepper = self._stepper
        path = stepper._resolve_group(group, self.group) if group is not None else None
        return [
            MesaSolver._get_view_of(agent)
            for agent in self._agent.cell.agents
            if agent is not self._agent and (path is None or stepper._get_spec_of(agent).path == path)
        ]

    def get_regions_here(self, layer: str) -> list[int]:
        """Return the indices of the regions of a layer that contain the agent.

        Regions with the agent on their boundaries count too.

        Args:
            layer:
                The name of the layer.

        Returns:
            indices:
                The indices of the regions, in increasing order.

        Raises:
            SimulationError:
                The space has no such layer.
        """
        hits = self._stepper._get_index_of(layer).query(Point(self.location), predicate='intersects')
        return sorted(int(index) for index in hits)

    def move_to(self, location: object) -> None:
        """Move the agent to a cell, a node, or a position.

        It may also move to a uniform point inside a polygon.

        Args:
            location:
                The cell coordinates, network node, or position, or a polygon in a continuous space.

        Raises:
            SpaceError:
                The location does not fit the space.
        """
        space = self._stepper.problem_.agent_space
        if isinstance(space, ContinuousSpace):
            self._agent.position = self._stepper._get_position_of(location, space)
            return
        self._agent.cell = self._stepper._get_cell_at(location)

    def remove(self) -> None:
        """Remove the agent from the system.

        The agent is first checked to hold none of a conserved quantity.

        Raises:
            ConservationError:
                A state tagged with a quantity is not zero, so removing the agent would destroy it. Settle the
                holdings with transfers first.

            SimulationError:
                The agent's group is not variable, so its size cannot change.
        """
        spec = self._stepper._get_spec_of(self._agent)
        MesaSolver._check_variable(spec, 'remove')
        held = {
            name: value
            for name in self._stepper._quantities[spec.path]
            if (value := float(getattr(self._agent, name))) != 0.0
        }
        if self.alive and held:
            error_msg = (
                f'Agent {self.id} of {spec.path} is removed while holding {held}. Settle its holdings with transfers '
                'to other agents before removing it, so conserved quantities are not destroyed.'
            )
            raise ConservationError(error_msg)
        if self.alive:
            self._stepper._recount(self._agent, -1)
            self._agent.removed_ = True
            self._agent.remove()

    def create(self, group: str | None = None, location: object = None, **values: object) -> 'AgentView':
        """Create an agent of a group.

        By default, it is of the agent's own group at the agent's location.

        Args:
            group:
                The path of the group, or `None` for the agent's group.

            location:
                The location of the new agent, or `None` for the agent's location.

            **values:
                The initial values of states, with the element type's defaults for the others.

        Returns:
            agent:
                The new agent.

        Raises:
            SimulationError:
                The group is not variable, so its size cannot change.
        """
        stepper = self._stepper
        target = stepper._resolve_group(group, self.group) if group is not None else self.group
        place = location if location is not None else self.location
        return MesaSolver._get_view_of(stepper._create_agent(target, place, values))

    def get_referenced(self, state: str) -> 'AgentView | None':
        """Return the member of another group that a reference state of the agent names.

        The state names it by its id. A state declared with `StateSpec(reference=group)` holds the id of a member of the
        group, or a negative value for none, such as the bank of a customer or the employer of a worker.

        Args:
            state:
                The name of a reference state.

        Returns:
            agent:
                The member the state names, or `None` when it names none or the member was removed.

        Raises:
            KeyError:
                The state of the agent names no member of a group.
        """
        stepper = self._stepper
        path = stepper._get_reference_of(stepper._get_spec_of(self._agent), state)
        value = float(getattr(self._agent, state))
        if value < 0 or path is None:
            return None
        agent = stepper.members_.get(int(value))
        if agent is None or getattr(agent, 'removed_', False) or stepper._get_spec_of(agent).path != path:
            return None
        return MesaSolver._get_view_of(agent)

    def get_lags(self, state: str, window: int | None = None) -> npt.NDArray[np.float64]:
        """Return the past values of a state of an agent.

        They are the values at the ends of the last steps, the last one last. An agent remembers a state declared with
        lags, `StateSpec(lags=m)`: the agent stepper keeps its last `m` values in the companion states `<state>_lag1` to
        `<state>_lag<m>`, which rules may also read one by one. Before the first steps the lags hold their initial
        values, the `initial` of the spec or else the initial value of the state.

        Args:
            state:
                The name of a state with lags.

            window:
                The number of last values, or `None` for every kept value.

        Returns:
            values:
                The values, the oldest first.

        Raises:
            KeyError:
                The state of the agent keeps no lags.
        """
        spec = self._stepper._get_spec_of(self._agent)
        declaration = declarations_of(MesaSolver._get_agent_type(spec), VariableKind.STATE).get(state)
        if declaration is None or not declaration.lags:
            error_msg = (
                f'The agents of {spec.path} do not remember {state!r}. Declare the state with lags, such as '
                '`StateSpec(lags=3)`, to keep its last values.'
            )
            raise KeyError(error_msg)
        lags = declaration.lags if window is None else min(window, declaration.lags)
        return np.asarray([getattr(self, f'{state}_lag{index}') for index in range(lags, 0, -1)], dtype=float)


class MesaSolver(ExternalSolver):
    """Mesa backend of agent-based problems, one Mesa step per unit of time.

    The agents are the members of the elements and groups whose laws include `RuleLaw`s, and of the variable groups.
    Every step advances Mesa's clock by one, activates all agents in random order, or by the phases of the system's
    discrete time, and each agent runs the rules of its element type in declaration order. Interaction rules run once
    after the agents. Rules add and remove members of variable groups only. The class variable `supports` declares
    what the backend runs, which validation checks systems against: discrete time, no space or a lattice, network, or
    continuous space, rule, boundary, transfer, and matching laws, and elements and fixed and variable
    groups with any domain but Hilbert spaces. It runs a continuous space as a box of two or three sides.

    Attributes:
        system_:
            The Mesa system.

        members_:
            Every agent created in the run by its id, which reference states name.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element, GroupElement
        >>> from skcomplex.core.components.laws import RuleLaw
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.solvers import MesaSolver
        >>> from skcomplex.core.engine.runtime import simulate
        >>> class CounterElement(Element):
        ...     def __init__(self, label, count: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label)
        ...         self.count = count
        ...
        ...     def _get_laws(self, namespace):
        ...         return [RuleLaw(self.tick)]
        ...
        ...     def tick(self, agent, system):
        ...         agent.count += 1
        >>> system = CompositeSystem(elements=[GroupElement('counters', element=CounterElement, size=3)])
        >>> trajectory = simulate(system, until=4, solver=MesaSolver())
        >>> trajectory.get_agents('counters', 4).values['count'].tolist()
        [4.0, 4.0, 4.0]
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={DiscreteTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace},
            laws={RuleLaw, BoundaryLaw, TransferLaw, MatchLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.AGENT

    @classmethod
    def _reads_inputs(cls) -> bool:
        """Return whether the solver reads the inputs of other partitions."""
        return True

    def _seed(self) -> int:
        """Return a seed for Mesa drawn from the stepper's random number generator."""
        return int(self.rng_.integers(0, MAX_SEED))

    @classmethod
    def _get_name(cls) -> str:
        """Return the name of the library the solver wraps."""
        return 'Mesa'

    def _get_unsupported(self, problem: ProblemSpec) -> list[str]:
        """Return a continuous space that is not a box of two or three sides.

        It is the only problem Mesa does not run.
        """
        space = problem.agent_space
        if not isinstance(space, ContinuousSpace) or len(space.extent_) in CONTINUOUS_SIDES:
            return []
        description = (
            f'the continuous space {space.label!r} with the extent {space.extent_} and {space.n_dim_} dimensions, '
            'since the solver runs continuous spaces of two or three dimensions with an extent'
        )
        return [description]

    def _build(self, problem: ProblemSpec) -> None:
        """Build the Mesa system of an agent-based problem.

        Args:
            problem:
                The compiled problem.

        Raises:
            SolverError:
                The problem is not agent-based.
        """
        self.parameters_ = problem.initial_parameters.copy()
        self.inputs_: Mapping[str, object] = {}
        self.deposits_ = np.zeros(problem.states.size)
        self.reports_: dict[str, object] = {}
        self.reported_: dict[str, deque[object]] = {}
        self.storage_: dict[str, dict[str, object]] = {}
        self.buffer_: dict[tuple[mesa.Agent, str], object] | None = None
        self.round_ = 0
        self.visit_ = (0, 0)
        self.triangulation_: tuple[object, dict[int, list[mesa.Agent]]] = (None, {})
        self.observed_: dict[str, deque[object]] = {}
        self._compositions: dict[tuple[str, str], dict[Hashable, int]] = {}
        self._triangulations: dict[Polygon, tuple[Array, Array]] = {}
        self._live: Array | None = None
        self._state: Array = problem.initial_state
        self._evaluator: ObservableEvaluator | None = None
        self._populating = True
        self._specs = {spec.path: spec for spec in problem.agents}
        self._lags = {
            spec.path: {
                name: declaration.lags
                for name, declaration in declarations_of(self._get_agent_type(spec), VariableKind.STATE).items()
                if declaration.lags
            }
            for spec in problem.agents
        }
        self._quantities = {
            spec.path: [
                name
                for name, declaration in declarations_of(self._get_agent_type(spec), VariableKind.STATE).items()
                if declaration.quantity is not None
            ]
            for spec in problem.agents
        }
        self._classes: dict[type, CompiledAgentsSpec] = {}
        self._view_classes: dict[type, type[AgentView]] = {}
        self.members_: dict[int, mesa.Agent] = {}
        self.system_ = _AgentSystem(self._run_rules, self._seed())
        self.space_ = self._build_space(
            problem.agent_space,
            self.system_,
            sum(spec.size for spec in problem.agents),
        )
        roles = {item.path: item.roles for item in problem.interactions}
        self._views = {spec.path: SystemView(self, spec.path) for spec in problem.agents}
        self._rules: list[tuple[SystemView, RuleLaw]] = []
        self._member_rules: dict[str, list[tuple[SystemView, RuleLaw]]] = {}
        for owner, rule in problem.system_rules:
            if not isinstance(rule, RuleLaw):
                continue
            view = SystemView(self, owner, roles.get(owner))
            if rule.members is None:
                self._rules.append((view, rule))
                continue
            for path in view._get_paths(rule.members):
                self._member_rules.setdefault(path, []).append((view, rule))
        declared = self._get_declared_phases(problem.system) if problem.system is not None else []
        self._phases = self._get_phase_plan(declared, problem)
        owners = [(item.path, type(item.interaction)) for item in problem.interactions] + [
            (spec.path, spec.element.element if isinstance(spec.element, GroupElement) else type(spec.element))
            for spec in problem.agents
        ]
        for path, component in owners:
            for name in getattr(component, 'reports', ()):
                self.reports_[f'{path}.{name}'] = math.nan
        for spec in problem.agents:
            self._populate(spec)
        for spec in problem.agents:
            self._resolve_references(spec)
        self._populating = False
        self._collect()

    def _get_agent_class(self, spec: CompiledAgentsSpec) -> type:
        """Return the Mesa agent class of a group, created on first use."""
        for agent_class, known in self._classes.items():
            if known is spec:
                return agent_class
        space = self.problem_.agent_space
        base = (
            CellAgent
            if isinstance(space, LatticeSpace | NetworkSpace)
            else ContinuousSpaceAgent if space else mesa.Agent
        )

        def step(agent: mesa.Agent) -> None:
            """Run the rules of the agent's type and its member rules.

            They do not run if the agent was removed earlier.
            """
            view = self._get_view_of(agent)
            for system, rule in self._get_agent_rules(spec):
                if getattr(agent, 'removed_', False):
                    return
                rule.function(view, system)

        agent_class = type(f'{spec.path.replace("/", "_")}_agent', (base,), {'step': step})
        self._classes[agent_class] = spec
        self._view_classes[agent_class] = self._get_view_class(spec)
        return agent_class

    def _get_agent_rules(self, spec: CompiledAgentsSpec) -> list[tuple[SystemView, RuleLaw]]:
        """Return the rules of an agent with their views.

        Those of its element come first, then its member rules.
        """
        own = [(self._views[spec.path], rule) for rule in self._get_rules_of(spec.rules)]
        return own + self._member_rules.get(spec.path, [])

    def _place(self, agent: mesa.Agent, location: object) -> None:
        """Place a new agent at a location.

        Without one, it goes to a random place that respects the space's capacity.
        """
        space = self.problem_.agent_space
        if isinstance(space, ContinuousSpace):
            sides = len(space.extent_)
            point = location if location is not None else self.system_.rng.uniform(np.zeros(sides), space.extent_)
            agent.position = self._get_position_of(point, space)
        elif space is not None:
            cells = self._get_cells()
            if location is not None:
                agent.cell = self._get_cell_at(location)
            elif space.capacity is not None:
                agent.cell = cells.select_random_empty_cell()
            else:
                agent.cell = cells.all_cells.select_random_cell()

    def _get_triangles_of(self, polygon: Polygon) -> tuple[Array, Array]:
        """Return the corners of the constrained Delaunay triangles of a polygon.

        Their areas are returned too, and both are computed once.

        Args:
            polygon:
                The polygon.

        Returns:
            triangles:
                The corners of every triangle, of the shape `(n, 3, 2)`, and the areas of the triangles.

        Raises:
            SpaceError:
                The polygon is empty or invalid, so it has no triangles to draw a point from.
        """
        known = self._triangulations.get(polygon)
        if known is not None:
            return known
        if polygon.is_empty or not polygon.is_valid:
            error_msg = f'An agent is placed in an invalid polygon: {shapely.is_valid_reason(polygon)}. Repair it.'
            raise SpaceError(error_msg)
        triangles = shapely.get_parts(shapely.constrained_delaunay_triangles(polygon))
        corners = shapely.get_coordinates(triangles).reshape(-1, 4, 2)[:, :3]
        self._triangulations[polygon] = corners, shapely.area(triangles)
        return self._triangulations[polygon]

    def _draw_point(self, polygon: Polygon) -> Array:
        """Return a uniform point of a polygon.

        A triangle is drawn by its area, then a uniform point of the triangle.
        """
        corners, areas = self._get_triangles_of(polygon)
        rng = self.system_.rng
        first, second, third = corners[int(rng.choice(len(areas), p=areas / areas.sum()))]
        along, across = rng.random(2)
        if along + across > 1.0:
            along, across = 1.0 - along, 1.0 - across
        return first + along * (second - first) + across * (third - first)

    def _get_position_of(self, location: object, space: ContinuousSpace) -> Array:
        """Return the position a location of a continuous space names.

        A polygon names a point drawn inside it.

        Args:
            location:
                The coordinates of a position, or a polygon.

            space:
                The continuous space.

        Returns:
            position:
                The coordinates, one per dimension of the space.

        Raises:
            SpaceError:
                The location does not have one coordinate per dimension of the space.
        """
        position = self._draw_point(location) if isinstance(location, Polygon) else np.asarray(location, dtype=float)
        if position.shape != (space.n_dim_,):
            error_msg = (
                f'The location {location!r} has {position.size} coordinates, and the space has {space.n_dim_} '
                f'dimensions. Give {space.n_dim_} coordinates, or a polygon in a space of two dimensions.'
            )
            raise SpaceError(error_msg)
        return position

    def _get_cell_at(self, location: object) -> object:
        """Return the Mesa cell of the coordinates of a cell or a network node.

        Args:
            location:
                The cell coordinates or the network node.

        Returns:
            cell:
                The Mesa cell.

        Raises:
            SpaceError:
                The location is a polygon, which only a continuous space holds.
        """
        if isinstance(location, Polygon):
            error_msg = 'A polygon is a location of a continuous space. Give the coordinates of a cell or a node.'
            raise SpaceError(error_msg)
        return self._get_cells()[self._get_cell_key(location)]

    def _get_index_of(self, layer: str) -> STRtree:
        """Return the index of the regions of a layer of the continuous space.

        Args:
            layer:
                The name of the layer.

        Returns:
            index:
                The index, which holds the polygons of the layer in their order.

        Raises:
            SimulationError:
                The space has no such layer.
        """
        space = self.problem_.agent_space
        layers = space.regions_ if isinstance(space, ContinuousSpace) else {}
        if layer not in layers:
            error_msg = f'The space has no region layer {layer!r}. Its layers are: {", ".join(layers) or "none"}.'
            raise SimulationError(error_msg)
        return layers[layer]

    def _get_cells(self) -> OrthogonalMooreGrid | OrthogonalVonNeumannGrid | Network:
        """Return the Mesa grid or network the agents live on.

        Returns:
            cells:
                The Mesa grid or network.

        Raises:
            SimulationError:
                The agents live in a continuous space or in no space, which has no cells.
        """
        space = self.space_
        if not isinstance(space, OrthogonalMooreGrid | OrthogonalVonNeumannGrid | Network):
            error_msg = 'The agents live in no grid or network, which has no cells.'
            raise SimulationError(error_msg)
        return space

    def _resolve_group(self, group: str, scope: str = '') -> str:
        """Return the path of a group named relative to the system of a component.

        A group may also be named by its path.

        Args:
            group:
                The path of the group relative to the system that holds `scope`, or from the root.

            scope:
                The path of a group or an interaction whose system relative paths are resolved in.

        Returns:
            path:
                The path of the group.

        Raises:
            SimulationError:
                No group has that path or label.
        """
        prefix = scope.rpartition('/')[0]
        for candidate in (f'{prefix}/{group}' if prefix else group, group):
            if candidate in self._specs:
                return candidate
        error_msg = f'No agent group {group!r}. Available groups are: {", ".join(self._specs)}.'
        raise SimulationError(error_msg)

    def _create_agent(self, group: str, location: object, values: Mapping[str, object]) -> mesa.Agent:
        """Create an agent of a group with initial state values.

        It is placed at a location or a random place.

        Args:
            group:
                The path of the group.

            location:
                The location, a polygon for a uniform point inside it, or `None` for a random place.

            values:
                The initial values of states, with the element type's defaults for the others.

        Returns:
            agent:
                The Mesa agent.

        Raises:
            SimulationError:
                The group does not exist, or it is not variable, so its size cannot change.
        """
        spec = self._specs[self._resolve_group(group)]
        if not self._populating:
            self._check_variable(spec, 'create')
        agent_class = self._get_agent_class(spec)
        agent = (
            agent_class(self.space_, self.system_)
            if agent_class.__mro__[1] is ContinuousSpaceAgent
            else agent_class(self.system_)
        )
        vars(agent)['view_'] = self._view_classes[agent_class](agent, self)
        self.members_[int(agent.unique_id)] = agent
        defaults = self._get_agent_type(spec)
        for name in spec.states:
            value = values[name] if name in values else self._get_shared_default(defaults, name)
            setattr(agent, name, value)
        self._place(agent, location)
        self._recount(agent, 1)
        return agent

    def _populate(self, spec: CompiledAgentsSpec) -> None:
        """Create the initial agents of a group.

        They take their per-agent or shared initial values.
        """
        space = self.problem_.agent_space
        network_nodes = list(space.graph.nodes) if isinstance(space, NetworkSpace) else []
        group = spec.element if isinstance(spec.element, GroupElement) and spec.element.location is not None else None
        for index in range(spec.size):
            values = {name: self._get_member_of(spec, name, value, index) for name, value in spec.values.items()}
            location = None if group is None else self._get_cell_key(group._get_member_location(index, spec.size))
            if location is None and spec.size <= len(network_nodes):
                location = network_nodes[index]
            self._create_agent(spec.path, location, values)

    def _get_reference_of(self, spec: CompiledAgentsSpec, state: str) -> str | None:
        """Return the path of the group a reference state of a group names members of.

        Args:
            spec:
                The compiled agent group.

            state:
                The name of the state.

        Returns:
            path:
                The path of the referenced group, or `None` when the system has no such group, so the state names no
                member.

        Raises:
            KeyError:
                The state names no member of a group.
        """
        declaration = declarations_of(self._get_agent_type(spec), VariableKind.STATE).get(state)
        if declaration is None or declaration.reference is None:
            error_msg = (
                f'The state {state!r} of {spec.path} names no member of a group. Declare it with '
                '`StateSpec(reference=group)` to read the member it names.'
            )
            raise KeyError(error_msg)
        try:
            return self._resolve_group(declaration.reference, spec.path)
        except SimulationError:
            return None

    def _resolve_references(self, spec: CompiledAgentsSpec) -> None:
        """Turn the initial positions of the reference states of a group into ids.

        The ids are those of the members they name.

        Args:
            spec:
                The compiled agent group.

        Raises:
            SimulationError:
                An initial position is beyond the initial members of the referenced group.
        """
        declarations = declarations_of(self._get_agent_type(spec), VariableKind.STATE)
        for name, declaration in declarations.items():
            if declaration.reference is None:
                continue
            path = self._get_reference_of(spec, name)
            members = sorted(self._get_group_agents(path), key=lambda agent: agent.unique_id) if path else []
            ids = [agent.unique_id for agent in members]
            for agent in self._get_group_agents(spec.path):
                position = int(float(getattr(agent, name)))
                if position >= len(ids):
                    error_msg = (
                        f'The state {name!r} of {spec.path} names the member at position {position} of '
                        f'{path or declaration.reference}, which has {len(ids)} members.'
                    )
                    raise SimulationError(error_msg)
                setattr(agent, name, float(ids[position]) if position >= 0 else -1.0)

    def _get_spec_of(self, agent: mesa.Agent) -> CompiledAgentsSpec:
        """Return the group of a Mesa agent created by the stepper.

        Args:
            agent:
                The Mesa agent.

        Returns:
            spec:
                The compiled agent group.
        """
        return self._classes[type(agent)]

    def _get_agents_of(self, group: str) -> list[AgentView] | None:
        """Return the views of the alive agents of a group.

        It returns `None` when the system has no such group.
        """
        if group not in self._specs:
            return None
        return [self._get_view_of(agent) for agent in self._get_group_agents(group)]

    def _get_delaunay_neighbors(self, agent: mesa.Agent, group: str, over: DelaunayNeighborhood) -> list[mesa.Agent]:
        """Return the agents of a group that share a Delaunay edge with an agent.

        The triangulation is that of the positions of the agents.

        The triangulation is kept for the step while the number of agents of the group stays the same, so it is
        computed again in a step only after agents are created or removed, and an agent removed in the step is left out.

        Args:
            agent:
                The Mesa agent.

            group:
                The path of the group.

            over:
                The Delaunay neighbourhood.

        Returns:
            agents:
                The neighbouring Mesa agents, with the agent first for a neighbourhood with its centre.

        Raises:
            SpaceError:
                The agent space is not continuous.
        """
        space = self.space_
        if not isinstance(space, MesaContinuousSpace):
            error_msg = 'Delaunay neighbours triangulate positions, so they need a continuous space.'
            raise SpaceError(error_msg)
        members = self._get_group_agents(group)
        key = (float(self.system_.time), self.visit_, group, len(members))
        if self.triangulation_[0] != key:
            positions = np.asarray([MesaSolver._get_location_of(item) for item in members], dtype=float)
            periods = tuple(float(side) for side in space.size) if space.torus else None
            readers, others = MetricSearch._get_delaunay(positions.reshape(len(members), -1), periods)
            table: dict[int, list[mesa.Agent]] = {item.unique_id: [] for item in members}
            for reader, other in zip(readers.tolist(), others.tolist(), strict=True):
                table[members[reader].unique_id].append(members[other])
            self.triangulation_ = (key, table)
        found = [
            item for item in self.triangulation_[1].get(agent.unique_id, []) if not getattr(item, 'removed_', False)
        ]
        return [agent, *found] if over.center else found

    def _get_group_agents(self, group: str) -> list[mesa.Agent]:
        """Return the alive Mesa agents of a group.

        Args:
            group:
                The path of the group.

        Returns:
            agents:
                The Mesa agents.
        """
        for agent_class, spec in self._classes.items():
            if spec.path == group:
                return list(self.system_.agents_by_type.get(agent_class, []))
        return []

    def _push_parameters(self, p: Array) -> None:
        """Make new parameter values visible to the rules."""
        self.parameters_ = p.copy()

    def _get_composition(self, path: str, state: str) -> dict[Hashable, int]:
        """Return the counts of the values of a state among the agents of a group.

        They are kept from their first reading.

        Args:
            path:
                The path of the group.

            state:
                The name of the state.

        Returns:
            counts:
                The number of agents of the group by value of the state.

        Raises:
            SimulationError:
                The agents of the group have no such state.
        """
        states = self._specs[path].states
        if state not in states:
            error_msg = f'The agents of {path} have no state {state!r}. Count one of: {", ".join(states)}.'
            raise SimulationError(error_msg)
        counts = self._compositions.get((path, state))
        if counts is None:
            counts = {}
            for agent in self._get_group_agents(path):
                value = MesaSolver._get_counted(getattr(agent, state))
                counts[value] = counts.get(value, 0) + 1
            self._compositions[path, state] = counts
        return counts

    def _recount(self, agent: mesa.Agent, change: int, names: Iterable[str] | None = None) -> None:
        """Add a change to the counts of the values an agent holds.

        The change covers the states named or every counted state.
        """
        if not self._compositions:
            return
        path = self._get_spec_of(agent).path
        for (group, state), counts in self._compositions.items():
            if group == path and (names is None or state in names):
                value = MesaSolver._get_counted(getattr(agent, state))
                counts[value] = counts.get(value, 0) + change

    def _write(self, agent: mesa.Agent, name: str, value: object) -> None:
        """Set a state of an agent and keep the counts of its values."""
        self._recount(agent, -1, (name,))
        setattr(agent, name, value)
        self._recount(agent, 1, (name,))

    def _observe(self) -> None:
        """Add the scalar values of the observables of the run to their histories.

        They are taken at the end of a step.
        """
        if not self.problem_.observables:
            return
        if self._evaluator is None:
            self._evaluator = ObservableEvaluator(self.problem_).prepare()
        values = self._evaluator.evaluate(float(self.system_.time), self._state, self.parameters_, self._get_agents_of)
        for name, value in values.items():
            array = value if isinstance(value, np.ndarray) else None
            history = self.observed_.setdefault(name, deque(maxlen=HISTORY_LENGTH))
            if isinstance(value, numbers.Number) or (
                array is not None and array.ndim == 0 and np.issubdtype(array.dtype, np.number)
            ):
                history.append(float(np.real(np.asarray(value))))
            elif isinstance(value, Hashable):
                history.append(value)

    def _run_rules(self) -> None:
        """Run the phases due in a step.

        Without phases, the agents are activated in random order and the system rules run once.
        """
        if self._phases is None:
            self.system_.agents.shuffle_do('step')
            for system, rule in self._rules:
                rule.function(system)
        else:
            for phase in self._phases:
                if (round(self.system_.time) - phase.offset) % phase.period:
                    continue
                for index in range(phase.repeat):
                    self.round_ = index
                    self._run_phase(phase)
            self.round_ = 0
        self._remember()
        self._observe()

    def _get_phase_paths(self, phase: PhaseSpec) -> list[str]:
        """Return the paths of the groups a phase activates, those it names or every group."""
        return [self._resolve_group(group) for group in phase.groups or [spec.path for spec in self.problem_.agents]]

    def _get_ordered_agents(self, phase: PhaseSpec) -> list[mesa.Agent]:
        """Return the agents of a phase in the order of its activation policy."""
        if phase.activation == 'sequential':
            return sorted(self.system_.agents, key=lambda agent: agent.unique_id)
        if phase.activation == 'by-type':
            paths = self._get_phase_paths(phase)
            ordered: list[mesa.Agent] = []
            for path in paths:
                members = sorted(self._get_group_agents(path), key=lambda agent: agent.unique_id)
                ordered += [members[index] for index in self.system_.rng.permutation(len(members))]
            return ordered
        agents = sorted(self.system_.agents, key=lambda agent: agent.unique_id)
        return [agents[index] for index in self.system_.rng.permutation(len(agents))]

    def _activate(self, agent: mesa.Agent, phase: PhaseSpec) -> None:
        """Run the rules of an agent in a phase while it lives.

        Those of its element run first, then its member rules.
        """
        view = self._get_view_of(agent)
        for system, rule in self._get_agent_rules(self._get_spec_of(agent)):
            if getattr(agent, 'removed_', False):
                return
            if self._get_phase_of(rule) == phase.name:
                rule.function(view, system)

    def _visit(self, phase: PhaseSpec) -> None:
        """Activate agents of a phase drawn with replacement.

        There are `visits` times as many as there are agents at its start.
        """
        paths = self._get_phase_paths(phase)
        count = int(sum(len(self._get_group_agents(path)) for path in paths) * phase.visits)
        try:
            for visit in range(count):
                agents = [agent for path in paths for agent in self._get_group_agents(path)]
                if not agents:
                    return
                self.visit_ = (visit, count)
                self._activate(agents[int(self.system_.rng.integers(len(agents)))], phase)
        finally:
            self.visit_ = (0, 0)

    def _run_phase(self, phase: PhaseSpec) -> None:
        """Run the agent rules of a phase under its activation policy.

        Then the system rules of the phase run.
        """
        if phase.activation == 'simultaneous':
            self.buffer_ = {}
        try:
            if phase.activation == 'visits':
                self._visit(phase)
            else:
                for agent in self._get_ordered_agents(phase):
                    self._activate(agent, phase)
        finally:
            buffer, self.buffer_ = self.buffer_, None
        for (agent, name), value in (buffer or {}).items():
            self._write(agent, name, value)
        for system, rule in self._rules:
            if self._get_phase_of(rule) == phase.name:
                rule.function(system)

    def _remember(self) -> None:
        """Shift the past values of the states with lags of every agent.

        This happens at the end of a step.
        """
        for agent in self.system_.agents:
            for name, lags in self._lags[self._get_spec_of(agent).path].items():
                for index in range(lags, 1, -1):
                    setattr(agent, f'{name}_lag{index}', getattr(agent, f'{name}_lag{index - 1}'))
                setattr(agent, f'{name}_lag1', getattr(agent, name))

    def _set_parameter(self, name: str, value: float) -> None:
        """Set a parameter during a step.

        The rules read it at once and the trajectory records it.

        Args:
            name:
                The dotted name of the parameter.

            value:
                The new value.
        """
        _, indices = self.problem_.parameters.locate(name)
        self.parameters_[indices] = value
        if self._live is not None:
            self._live[indices] = value

    def _get_observable(self, name: str) -> object:
        """Return the current value of an observable of the problem.

        Args:
            name:
                The path of the observable.

        Returns:
            value:
                Its value for the current agents and parameters.

        Raises:
            SimulationError:
                The problem has no such observable.
        """
        if self._evaluator is None:
            self._evaluator = ObservableEvaluator(self.problem_).prepare()
        values = self._evaluator.evaluate(float(self.system_.time), self._state, self.parameters_, self._get_agents_of)
        if name not in values:
            error_msg = f'No observable named {name!r}. Available observables are: {", ".join(values) or "none"}.'
            raise SimulationError(error_msg)
        return values[name]

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Run one Mesa step per integer time from `t` up to `until`.

        Args:
            t:
                The current time.

            x:
                The current state vector, empty for agent-based problems.

            p:
                The parameter vector.

            until:
                The time to advance to.

        Returns:
            state:
                The unchanged state vector.
        """
        if not np.array_equal(p, self.parameters_):
            self._push_parameters(p)
        self._live, self._state = p, x
        for step in range(math.floor(t + 1e-9), math.floor(until + 1e-9)):
            del step
            self.system_.step()
        return x

    def _collect(self) -> None:
        """Set the agents of every group and the reporters at the current step."""
        groups, agents = {}, {}
        for spec in self.problem_.agents:
            agents[spec.path] = self._get_group_agents(spec.path)
            groups[spec.path] = self._get_snapshot_of(
                agents[spec.path],
                {name: self._get_attribute(name) for name in spec.states},
                spec,
            )
        reporters = {f'{path}.count': float(len(snapshot.ids)) for path, snapshot in groups.items()}
        super()._collect()
        self.snapshot_ = AgentStateSpec(groups, {**reporters, **cast(dict[str, float], self.reports_)})
        self.agents_ = {path: [self._get_view_of(agent) for agent in members] for path, members in agents.items()}

    @staticmethod
    def _build_space(space: BaseSpace | None, system: mesa.Model, n_agents: int) -> MesaSpace | None:
        """Return the Mesa space of a framework space."""
        if isinstance(space, LatticeSpace):
            grid = OrthogonalVonNeumannGrid if space.metric_ == 'manhattan' else OrthogonalMooreGrid
            return grid(space.shape_[:2], torus=space.periodic_, capacity=space.capacity, random=system.random)
        if isinstance(space, NetworkSpace):
            return Network(space.graph, capacity=space.capacity, random=system.random, layout=None)
        if isinstance(space, ContinuousSpace):
            bounds = [[0.0, side] for side in space.extent_]
            room = max(n_agents, MIN_CONTINUOUS_ROOM)
            return MesaContinuousSpace(bounds, torus=space.periodic_, random=system.random, n_agents=room)
        return None

    @staticmethod
    def _get_rules_of(laws: Iterable[BaseLaw]) -> list[RuleLaw]:
        """Return the rules among laws.

        The compiled agents and the system rules hold them alone.
        """
        return [law for law in laws if isinstance(law, RuleLaw)]

    @staticmethod
    def _get_phase_of(rule: RuleLaw) -> str:
        """Return the name of the phase a rule runs in."""
        return rule.phase or DEFAULT_PHASE

    @staticmethod
    def _get_declared_phases(system: CompositeSystem) -> list[PhaseSpec]:
        """Return the phases of the schedules of a system and its nested systems.

        The first phase of every name is kept.
        """
        time = system.time
        schedule = time.prepare().schedule_ if isinstance(time, DiscreteTime) else None
        phases = list(schedule.phases) if isinstance(schedule, PhasesScheduleSpec) else []
        for nested in system.systems:
            phases += MesaSolver._get_declared_phases(nested)
        kept: dict[str, PhaseSpec] = {}
        for phase in phases:
            kept.setdefault(phase.name, phase)
        return list(kept.values())

    @staticmethod
    def _get_phase_plan(phases: Sequence[PhaseSpec], problem: ProblemSpec) -> list[PhaseSpec] | None:
        """Return the phases of a step with the default phase.

        It returns `None` when no phase is declared or named.
        """
        rules = MesaSolver._get_rules_of(
            [*(rule for spec in problem.agents for rule in spec.rules), *(rule for _, rule in problem.system_rules)],
        )
        if not phases and all(rule.phase is None for rule in rules):
            return None
        names = [phase.name for phase in phases]
        plan = list(phases) if DEFAULT_PHASE in names else [PhaseSpec(DEFAULT_PHASE), *phases]
        unknown = sorted({MesaSolver._get_phase_of(rule) for rule in rules} - {phase.name for phase in plan})
        if unknown:
            error_msg = (
                f'Rules run in the phases {", ".join(unknown)}, which the system does not declare. Declared phases '
                f'are: {", ".join(names) or "none"}.'
            )
            raise SimulationError(error_msg)
        return plan

    @staticmethod
    def _get_attribute(name: str) -> Callable[[mesa.Agent], object]:
        """Return a function that reads an attribute of an agent."""
        return lambda agent: getattr(agent, name)

    @staticmethod
    def _get_snapshot_of(
        agents: list[mesa.Agent],
        reporters: Mapping[str, Callable[[mesa.Agent], object]],
        spec: CompiledAgentsSpec,
    ) -> AgentSnapshotSpec:
        """Return the ids, reported values, and locations of agents.

        A state of objects gives one value per agent.
        """
        agents = sorted(agents, key=lambda agent: agent.unique_id)
        ids = np.array([agent.unique_id for agent in agents], dtype=np.int64)
        values = {
            name: (
                np.fromiter(map(reporter, agents), dtype=object, count=len(agents))
                if name in spec.objects
                else np.asarray([reporter(agent) for agent in agents])
            )
            for name, reporter in reporters.items()
        }
        locations = [MesaSolver._get_location_of(agent) for agent in agents]
        has_locations = any(location is not None for location in locations)
        return AgentSnapshotSpec(ids, values, np.asarray(locations) if has_locations else None, spec.roles)

    @staticmethod
    def _get_shared_default(element_type: type, name: str) -> object:
        """Return the default of a state of an element type."""
        return default_of(element_type, name)

    @staticmethod
    def _get_state_property(name: str) -> property:
        """Return the property of a state of agents.

        The property sets the agent at the end of a simultaneous phase.
        """

        def read(view: AgentView) -> object:
            return getattr(view._agent, name)

        def write(view: AgentView, value: object) -> None:
            buffer = view._stepper.buffer_
            if buffer is not None:
                buffer[view._agent, name] = value
                return
            view._stepper._write(view._agent, name, value)

        return property(read, write, doc=f'State {name} of the agent.')

    @staticmethod
    def _get_parameter_property(name: str, entry: str) -> property:
        """Return the read-only property of a parameter of the group of agents."""

        def read(view: AgentView) -> float:
            stepper = view._stepper
            return float(np.ravel(stepper.problem_.parameters.read(stepper.parameters_, entry))[0])

        return property(read, doc=f'Parameter {name} of the group of the agent.')

    @staticmethod
    def _get_view_class(spec: CompiledAgentsSpec) -> type[AgentView]:
        """Return the class of the views of a group's agents.

        The class has a property per state and parameter.
        """
        names: dict[str, object] = {
            name: MesaSolver._get_parameter_property(name, entry)
            for name, entry in spec.parameters.items()
            if name not in spec.states
        }
        names.update({name: MesaSolver._get_state_property(name) for name in spec.states})
        names['__slots__'] = ()
        base = MesaSolver._get_agent_type(spec)._get_view() or AgentView
        if not issubclass(base, AgentView):
            error_msg = f'The view of {spec.path} is {base.__name__}, which does not derive from AgentView.'
            raise ValidationError(error_msg)
        return type(f'{spec.path.replace("/", "_")}_view', (base,), names)

    @staticmethod
    def _get_counted(value: object) -> Hashable:
        """Return the key that counts a value of a state.

        It is a float for a number, and an object as it is.
        """
        if isinstance(value, numbers.Real):
            return float(value)
        if not isinstance(value, Hashable):
            error_msg = f'A composition counts hashable values, but an agent holds {value!r}. Give it as a tuple.'
            raise SimulationError(error_msg)
        return value

    @staticmethod
    def _get_member_of(spec: CompiledAgentsSpec, name: str, value: object, index: int) -> object:
        """Return the initial value of one agent.

        It comes from a list of one object per agent, or from one shared value.
        """
        if name not in spec.objects:
            return MesaSolver._get_member_value(value, index, spec.size)
        return value[index] if isinstance(value, list | np.ndarray) and len(value) == spec.size else value

    @staticmethod
    def _get_member_value(value: object, index: int, size: int) -> object:
        """Return the initial value of one agent from a per-agent array or a shared value."""
        array = np.asarray(value)
        if array.ndim and array.shape[0] == size:
            item = array[index]
            return item.item() if np.ndim(item) == 0 else item
        return value

    @staticmethod
    def _get_agent_type(spec: CompiledAgentsSpec) -> type[BaseElement]:
        """Return the element type of a compiled agent element or group."""
        return spec.element.element if isinstance(spec.element, GroupElement) else type(spec.element)

    @staticmethod
    def _get_cell_key(location: object) -> object:
        """Return a location as a hashable cell key.

        The key is a tuple of grid coordinates or a network node.
        """
        if isinstance(location, np.ndarray | list | tuple):
            values = np.asarray(location).tolist()
            return tuple(values) if isinstance(values, list) else values
        return location.item() if isinstance(location, np.generic) else location

    @staticmethod
    def _check_variable(spec: CompiledAgentsSpec, action: str) -> None:
        """Raise an error when a rule changes the size of a fixed agent element or group.

        Groups that are not variable have a fixed size.
        """
        if isinstance(spec.element, GroupElement) and spec.element.variable:
            return
        error_msg = (
            f'A rule tries to {action} an agent of {spec.path}, whose size is fixed. Declare the group with '
            '`variable=True`, such as `GroupElement(label, element=..., size=..., variable=True)`, for its members '
            'to be created and removed during a run.'
        )
        raise SimulationError(error_msg)

    @staticmethod
    def _get_location_of(agent: mesa.Agent) -> object:
        """Return the location of a Mesa agent.

        It is grid coordinates, a network node, a position, or `None`.
        """
        cell = getattr(agent, 'cell', None)
        if cell is not None:
            return cell.coordinate
        position = getattr(agent, 'position', None)
        if position is not None:
            return tuple(np.asarray(position, dtype=float))
        return getattr(agent, 'pos', None)

    @staticmethod
    def _get_view_of(agent: mesa.Agent) -> AgentView:
        """Return the view of an agent.

        The stepper creates the view with the agent.
        """
        view = vars(agent)['view_']
        if not isinstance(view, AgentView):
            error_msg = f'The agent holds a {type(view).__name__}, not the view the stepper creates with it.'
            raise TypeError(error_msg)
        return view
