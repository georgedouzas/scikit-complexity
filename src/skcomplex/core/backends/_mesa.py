"""Run agent-based problems on the Mesa backend, one Mesa step per unit of time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, cast

import mesa
import networkx as nx
import numpy as np
import numpy.typing as npt
from mesa.discrete_space import CellAgent, Network, OrthogonalMooreGrid, OrthogonalVonNeumannGrid
from mesa.experimental.continuous_space import ContinuousSpace as MesaContinuousSpace
from mesa.experimental.continuous_space import ContinuousSpaceAgent

from ..base import (
    ComponentKind,
    ConservationError,
    DomainKind,
    Form,
    LawKind,
    SimulationError,
    SpaceError,
    SpaceKind,
    Supports,
    TimeKind,
    VariableKind,
    declarations_of,
)
from ..compile import CompiledAgents, ObservableEvaluator, Problem, Vector
from ..components import Group
from ..laws import Rule
from ..solvers import Stepper, register_solver
from ..solvers._agents import AgentSnapshot, AgentState
from ..space import ContinuousSpace, LatticeSpace, NetworkSpace, Phase, Space
from ..space._phases import DEFAULT_PHASE
from ..space._time import DiscreteTime

MAX_SEED = 2**31 - 1
HISTORY_LENGTH = 10_000


def _cell_key(location: object) -> object:
    """Return a location as a hashable cell key, a tuple of grid coordinates or a network node."""
    if isinstance(location, np.ndarray | list | tuple):
        values = np.asarray(location).tolist()
        return tuple(values) if isinstance(values, list) else values
    return location.item() if isinstance(location, np.generic) else location


def _location_of(agent: mesa.Agent) -> object:
    """Return the location of a Mesa agent: grid coordinates, a network node, a position, or `None`."""
    cell = getattr(agent, 'cell', None)
    if cell is not None:
        return cell.coordinate
    position = getattr(agent, 'position', None)
    if position is not None:
        return tuple(np.asarray(position, dtype=float))
    return getattr(agent, 'pos', None)


def _agent_type(spec: CompiledAgents) -> Any:  # noqa: ANN401
    """Return the element type of a compiled agent element or group."""
    return spec.element.element if isinstance(spec.element, Group) else type(spec.element)


class ModelView:
    """Model as rules see it: parameters, random numbers, time, agents, and the space.

    Every rule receives the view scoped to the interaction or agent group that owns it, so the rule reads the owner's
    parameters by name and the groups bound to the owner's roles by role, also when its system is nested in another.

    Args:
        stepper:
            The stepper that runs the model.

        owner:
            The path of the interaction or agent group whose rules receive the view, or `''` for no owner.

        roles:
            The paths of the elements and groups the owner's roles bind to, by role.
    """

    def __init__(
        self,
        stepper: 'MesaStepper',
        owner: str = '',
        roles: Mapping[str, str | tuple[str, ...]] | None = None,
    ) -> None:
        self._stepper = stepper
        self._owner = owner
        self._roles = dict(roles or {})

    def _paths(self, name: str) -> list[str]:
        """Return the paths of the groups a role of the owner binds to, or of a group by relative or absolute path."""
        bound = self._roles.get(name)
        if bound is not None:
            return [bound] if isinstance(bound, str) else list(bound)
        try:
            return [self._stepper.resolve_group(name, self._owner)]
        except SimulationError as error:
            roles, owner = ', '.join(self._roles) or 'none', self._owner or 'the model'
            error_msg = f'No role or agent group named {name!r} for {owner}. Its roles are: {roles}. {error}'
            raise SimulationError(error_msg) from error

    def parameter(self, name: str) -> float:
        """Return the current value of a parameter of the owner, or of one component as `name[i]`.

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
            names = [item.name[len(prefix) :] for item in layout if item.name.startswith(prefix)]
            error_msg = (
                f'{self._owner or "The model"} has no parameter {name!r}. Its parameters are: '
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
            for name, value in stepper.problem_.parameters.values(stepper.parameters_).items()
        }

    @property
    def inputs(self) -> dict[str, Any]:
        """Values of the states of other partitions at the last synchronisation point, by dotted name."""
        return self._stepper.inputs_

    @property
    def rng(self) -> np.random.Generator:
        """Random number generator of the model, seeded by the simulation's random state."""
        return self._stepper.model_.rng

    @property
    def time(self) -> float:
        """Time of Mesa's clock, which is `k` during the step that produces frame `k`."""
        return float(self._stepper.model_.time)

    @property
    def dimensions(self) -> tuple[float, ...] | None:
        """Extent of the agent space per axis, cells of a grid or lengths of a continuous space, `None` otherwise."""
        space = self._stepper.space_
        if isinstance(space, MesaContinuousSpace):
            return (float(space.width), float(space.height))
        if isinstance(space, OrthogonalMooreGrid | OrthogonalVonNeumannGrid):
            return tuple(int(item) for item in space.dimensions)
        return None

    @property
    def torus(self) -> bool:
        """Whether the agent space wraps around at its edges."""
        return bool(getattr(self._stepper.space_, 'torus', False))

    def agents(self, name: str) -> list['AgentView']:
        """Return the agents of a group bound to a role of the owner, or of a group by relative or absolute path.

        Args:
            name:
                A role of the owner, the path of a group relative to the owner's system, or the path of a group.

        Returns:
            agents:
                The agents alive in the group, or in every group of a role that binds several.

        Raises:
            SimulationError:
                No role or group has that name.
        """
        stepper = self._stepper
        return [AgentView(agent, stepper) for path in self._paths(name) for agent in stepper.group_agents(path)]

    def count(self, name: str) -> int:
        """Return the number of agents of a group bound to a role of the owner, or of a group by path.

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
        return sum(len(self._stepper.group_agents(path)) for path in self._paths(name))

    def _site(self, name: str, cell: object) -> int:
        """Return the member index of a field state's site at a cell or network node."""
        problem = self._stepper.problem_
        if name not in problem.states:
            names = [entry.name for entry in problem.states if entry.members is not None]
            error_msg = f'No field state named {name!r}. Available group states are: {", ".join(names)}.'
            raise KeyError(error_msg)
        shape = problem.lattice_shape(name)
        if shape is not None:
            return int(np.ravel_multi_index(tuple(int(item) for item in np.atleast_1d(cast('Any', cell))), shape))
        owner = name.rpartition('.')[0]
        element = next(item for item in problem.elements if item.path == owner)
        space = getattr(element.element, 'space', None)
        if space is None or not hasattr(space, 'graph'):
            error_msg = f'{name} is not a state of a field on a lattice or a network.'
            raise KeyError(error_msg)
        return list(space.graph.nodes).index(cell)

    def field(self, name: str, cell: object) -> float:
        """Return the value of a field at a cell, as it was at the last synchronisation point.

        Args:
            name:
                The dotted path of the field state, such as `ground/pheromone.level`.

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
        site = self._site(name, cell)
        values = self._stepper.inputs_.get(name)
        if values is None:
            error_msg = f'The agents read the field {name}, which no other partition provides. Run it in co-simulation.'
            raise KeyError(error_msg)
        return float(np.ravel(values)[site])

    def deposit(self, name: str, cell: object, amount: float) -> None:
        """Add an amount to a field at a cell at the next synchronisation point.

        Args:
            name:
                The dotted path of the field state, such as `ground/pheromone.level`.

            cell:
                The lattice coordinates of the cell or a network node.

            amount:
                The amount to add.

        Raises:
            KeyError:
                The name is not a state of a field.
        """
        entry = self._stepper.problem_.states.entry(name) if name in self._stepper.problem_.states else None
        site = self._site(name, cell)
        self._stepper.deposits_[entry.start + site * entry.columns] += amount  # type: ignore[union-attr]

    def random_empty_cell(self) -> object:
        """Return the coordinates of a random empty cell of the grid or network.

        Returns:
            location:
                The cell coordinates or the network node.
        """
        return self._stepper.space_.select_random_empty_cell().coordinate

    def create(self, group: str, location: object = None, **values: object) -> 'AgentView':
        """Create an agent of a group, at a location or at a random place.

        Args:
            group:
                The path of the group.

            location:
                The cell coordinates, network node, or position, or `None` for a random place.

            **values:
                The initial values of states, with the element type's defaults for the others.

        Returns:
            agent:
                The new agent.

        Raises:
            SimulationError:
                The group is not variable, so its size cannot change.
        """
        return AgentView(self._stepper.create_agent(group, location, values), self._stepper)

    @property
    def round(self) -> int:
        """Index of the current round of a repeated phase, `0` outside repeated phases."""
        return self._stepper.round_

    @property
    def storage(self) -> dict[str, Any]:
        """Mutable mapping private to the owner that persists across the steps of one run, such as an order book."""
        return self._stepper.storage_.setdefault(self._owner, {})

    def _report_name(self, name: str) -> str:
        """Return the reporter name of a name of the owner."""
        return f'{self._owner}.{name}' if self._owner else name

    def report(self, name: str, value: float) -> None:
        """Publish a value of the owner as the reporter `<owner>.<name>` and add it to the reporter's history.

        The owner declares the names it reports in its class variable `reports`, so every frame records them, `NaN`
        before the first report.

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
                f'{self._owner or "The model"} reports {name!r}, which it does not declare. List it in the class '
                'variable `reports` of the interaction or element type.'
            )
            raise SimulationError(error_msg)
        self._stepper.reports_[full] = float(value)
        self._stepper.reported_.setdefault(full, deque(maxlen=HISTORY_LENGTH)).append(float(value))

    def history(self, name: str, window: int | None = None) -> npt.NDArray[np.float64]:
        """Return the past values of a reporter, the last one last.

        Args:
            name:
                The name of a value the owner reports, or the full name of any reporter, such as `bar.attendance`.

            window:
                The number of last values, or `None` for every recorded value.

        Returns:
            values:
                The values, at most `window` of them.
        """
        history = self._stepper.reported_
        values = history.get(self._report_name(name), history.get(name, ()))
        array = np.asarray(values, dtype=float)
        return array if window is None else array[len(array) - min(window, len(array)) :]

    def observable(self, name: str) -> Any:  # noqa: ANN401
        """Return the current value of an observable of the system, such as an aggregate of a group.

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
        return self._stepper.observable(name)

    def set_parameter(self, name: str, value: float) -> None:
        """Set a parameter of any group or interaction during the run, recorded in the trajectory like any parameter.

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
        full = name if name in layout else f'{self._owner}.{name}'
        if full not in layout:
            error_msg = f'No parameter named {name!r} to set. Available names are: {", ".join(layout.names)}.'
            raise SimulationError(error_msg)
        self._stepper.set_parameter(full, value)


class AgentView:
    """Agent as rules see it: its states as attributes, its group's parameters, its neighbours, and its location.

    Args:
        agent:
            The Mesa agent.

        stepper:
            The stepper that runs the model.
    """

    __slots__ = ('_agent', '_stepper')

    def __init__(self, agent: mesa.Agent, stepper: 'MesaStepper') -> None:
        object.__setattr__(self, '_agent', agent)
        object.__setattr__(self, '_stepper', stepper)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Return a state of the agent or a parameter of its group."""
        spec = self._stepper.spec_of(self._agent)
        if name in spec.states:
            return getattr(self._agent, name)
        if name in spec.parameters:
            stepper = self._stepper
            return float(np.ravel(stepper.problem_.parameters.read(stepper.parameters_, spec.parameters[name]))[0])
        error_msg = f'Agents of {spec.path} have no state or parameter {name!r}.'
        raise AttributeError(error_msg)

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        """Set a state of the agent, at the end of the phase in a simultaneous phase."""
        if name not in self._stepper.spec_of(self._agent).states:
            error_msg = f'{name!r} is not a state of the agent, so it cannot be set.'
            raise AttributeError(error_msg)
        buffer = self._stepper.buffer_
        if buffer is not None:
            buffer[self._agent, name] = value
            return
        setattr(self._agent, name, value)

    def __eq__(self, other: object) -> bool:
        """Check whether two views show the same agent."""
        return isinstance(other, AgentView) and other._agent is self._agent

    def __hash__(self) -> int:
        """Return the hash of the agent's id."""
        return hash(self._agent.unique_id)

    @property
    def id(self) -> int:
        """Unique id of the agent."""
        return int(self._agent.unique_id)

    @property
    def group(self) -> str:
        """Path of the agent's group."""
        return self._stepper.spec_of(self._agent).path

    @property
    def group_label(self) -> str:
        """Label of the agent's group, the last part of its path."""
        return self.group.rpartition('/')[2]

    @property
    def location(self) -> object:
        """Grid coordinates, network node, or position of the agent."""
        return _location_of(self._agent)

    @property
    def alive(self) -> bool:
        """Whether the agent has not been removed."""
        return not getattr(self._agent, 'removed_', False)

    def displacement_to(self, other: 'AgentView | object') -> npt.NDArray[np.float64]:
        """Return the vector from the agent to another agent or a location, the shortest one on a torus.

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
        model = ModelView(self._stepper)
        dimensions = model.dimensions
        if dimensions is None:
            error_msg = (
                'Displacements need a grid or a continuous space, and the agents live on a network or in no space. '
                'Use `distance_to` for the number of hops on a network.'
            )
            raise SpaceError(error_msg)
        target = other.location if isinstance(other, AgentView) else other
        delta = np.asarray(target, dtype=float) - np.asarray(self.location, dtype=float)
        if not model.torus:
            return delta
        size = np.asarray(dimensions, dtype=float)
        return delta - size * np.round(delta / size)

    def distance_to(self, other: 'AgentView | object') -> float:
        """Return the distance to another agent or a location, the number of hops on a network.

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
        return float(np.linalg.norm(self.displacement_to(other)))

    def neighbor_cells(self) -> list[object]:
        """Return the coordinates of the cells or nodes around the agent's cell or node.

        Returns:
            locations:
                The neighbouring cell coordinates or network nodes.
        """
        return [cell.coordinate for cell in self._agent.cell.neighborhood]

    def neighbors(self, radius: float | None = None) -> list['AgentView']:
        """Return the agents in the cells or nodes around the agent, or within a radius in a continuous space.

        Args:
            radius:
                The distance within which agents are neighbours in a continuous space.

        Returns:
            agents:
                The neighbouring agents.
        """
        stepper = self._stepper
        if radius is not None:
            agents, _ = self._agent.get_neighbors_in_radius(radius)
            return [AgentView(agent, stepper) for agent in agents if agent is not self._agent]
        return [AgentView(agent, stepper) for agent in self._agent.cell.neighborhood.agents]

    def agents_here(self, group: str | None = None) -> list['AgentView']:
        """Return the other agents in the agent's cell or node, of one group or of every group.

        Args:
            group:
                The path of a group, or `None` for every group.

        Returns:
            agents:
                The other agents at the same place.
        """
        stepper = self._stepper
        path = stepper.resolve_group(group, self.group) if group is not None else None
        return [
            AgentView(agent, stepper)
            for agent in self._agent.cell.agents
            if agent is not self._agent and (path is None or stepper.spec_of(agent).path == path)
        ]

    def move_to(self, location: object) -> None:
        """Move the agent to a cell, a node, or a position.

        Args:
            location:
                The cell coordinates, network node, or position.
        """
        if isinstance(self._stepper.problem_.agent_space, ContinuousSpace):
            self._agent.position = np.asarray(location, dtype=float)
            return
        self._agent.cell = self._stepper.space_[_cell_key(location)]

    def remove(self) -> None:
        """Remove the agent from the model, after checking that it holds none of a conserved quantity.

        Raises:
            ConservationError:
                A state tagged with a quantity is not zero, so removing the agent would destroy it. Settle the
                holdings with transfers first.

            SimulationError:
                The agent's group is not variable, so its size cannot change.
        """
        spec = self._stepper.spec_of(self._agent)
        _check_variable(spec, 'remove')
        declared = declarations_of(_agent_type(spec), VariableKind.STATE)
        held = {
            name: value
            for name, declaration in declared.items()
            if declaration.quantity is not None and (value := float(getattr(self._agent, name))) != 0.0
        }
        if self.alive and held:
            error_msg = (
                f'Agent {self.id} of {spec.path} is removed while holding {held}. Settle its holdings with transfers '
                'to other agents before removing it, so conserved quantities are not destroyed.'
            )
            raise ConservationError(error_msg)
        if self.alive:
            self._agent.removed_ = True
            self._agent.remove()

    def create(self, group: str | None = None, location: object = None, **values: object) -> 'AgentView':
        """Create an agent of a group, by default the agent's own group at the agent's location.

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
        target = stepper.resolve_group(group, self.group) if group is not None else self.group
        place = location if location is not None else self.location
        return AgentView(stepper.create_agent(target, place, values), stepper)


class _AgentModel(mesa.Model):
    """Mesa model whose step runs the agent and model rules of a stepper, so Mesa's clock moves with the run."""

    def __init__(self, run_rules: Callable[[], None], rng: int) -> None:
        self._run_rules = run_rules
        super().__init__(rng=rng)

    def step(self) -> None:
        """Run the agent rules in random order and then the model rules."""
        self._run_rules()


type MesaSpace = OrthogonalMooreGrid | OrthogonalVonNeumannGrid | Network | MesaContinuousSpace


def _build_space(space: Space | None, model: mesa.Model, n_agents: int) -> MesaSpace | None:
    """Return the Mesa space of a framework space."""
    if isinstance(space, LatticeSpace):
        grid = OrthogonalVonNeumannGrid if space.neighborhood == 'von_neumann' else OrthogonalMooreGrid
        return grid((space.width, space.height), torus=space.torus, capacity=space.capacity, random=model.random)
    if isinstance(space, NetworkSpace):
        return Network(space.graph, capacity=space.capacity, random=model.random, layout=None)
    if isinstance(space, ContinuousSpace):
        bounds = [[0.0, space.width], [0.0, space.height]]
        return MesaContinuousSpace(bounds, torus=space.torus, random=model.random, n_agents=max(n_agents, 1))
    return None


class MesaStepper(Stepper):
    """Mesa backend of agent-based problems, one Mesa step per unit of time.

    The agents are the members of the elements and groups whose laws include `Rule` laws, and of the variable groups.
    Every step advances Mesa's clock by one, activates all agents in random order, or by the phases of the system's
    discrete time, and each agent runs the rules of its element type in declaration order. Interaction rules run once
    after the agents. Rules add and remove members of variable groups only. The class variable `supports` declares
    what the backend runs, which validation checks systems against: discrete time, no space or a lattice, network, or
    continuous space, rule, boundary, transfer, matching, and invariant laws, and elements and fixed and variable
    groups with any domain but Hilbert spaces.

    Attributes:
        model_:
            The Mesa model.

    Examples:
        >>> from skcomplex.core.base import state
        >>> from skcomplex.core.components import Element
        >>> from skcomplex.core.laws import Rule
        >>> from skcomplex.core.system import System
        >>> from skcomplex.core.runtime import Simulation
        >>> class Counter(Element):
        ...     count: float = state(0.0)
        ...
        ...     def laws(self, s):
        ...         return [Rule(self.tick)]
        ...
        ...     def tick(self, agent, model):
        ...         agent.count += 1
        >>> trajectory = Simulation(System(components=[Counter.group('counters', 3)])).run(until=4).trajectory_
        >>> trajectory.agents('counters', 4).values['count'].tolist()
        [4.0, 4.0, 4.0]
    """

    supports: ClassVar[Supports] = Supports(
        time={TimeKind.DISCRETE},
        space={SpaceKind.NONE, SpaceKind.LATTICE, SpaceKind.NETWORK, SpaceKind.CONTINUOUS},
        laws={LawKind.RULE, LawKind.BOUNDARY, LawKind.TRANSFER, LawKind.MATCH, LawKind.INVARIANT},
        domains=set(DomainKind) - {DomainKind.HILBERT},
        components={ComponentKind.ELEMENT, ComponentKind.GROUP, ComponentKind.VARIABLE_GROUP, ComponentKind.SYSTEM},
    )

    form: ClassVar[Form] = Form.AGENT
    reads_inputs: ClassVar[bool] = True

    def _seed(self) -> int:
        """Return a seed for Mesa drawn from the stepper's random number generator."""
        return int(self.rng_.integers(0, MAX_SEED))

    def initialize(self, problem: Problem, rng: np.random.Generator) -> None:
        """Build the Mesa model of an agent-based problem.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator that seeds Mesa.

        Raises:
            SolverError:
                The problem is not agent-based.
        """
        super().initialize(problem, rng)
        self.parameters_ = problem.initial_parameters.copy()
        self.inputs_: dict[str, Any] = {}
        self.deposits_ = np.zeros(problem.states.size)
        self.reports_: dict[str, float] = {}
        self.reported_: dict[str, deque[float]] = {}
        self.storage_: dict[str, dict[str, Any]] = {}
        self.buffer_: dict[tuple[mesa.Agent, str], Any] | None = None
        self.round_ = 0
        self._live: Vector | None = None
        self._state: Vector = problem.initial_state
        self._evaluator: ObservableEvaluator | None = None
        self._invariant_values: dict[tuple[int, str, str], float] = {}
        self._populating = True
        self._specs = {spec.path: spec for spec in problem.agents}
        self._classes: dict[type, CompiledAgents] = {}
        self.model_ = _AgentModel(self._run_rules, self._seed())
        self.space_: Any = _build_space(problem.agent_space, self.model_, sum(spec.size for spec in problem.agents))
        roles = {item.path: item.roles for item in problem.interactions}
        self._views = {spec.path: ModelView(self, spec.path) for spec in problem.agents}
        self._rules = [(ModelView(self, owner, roles.get(owner)), rule) for owner, rule in problem.model_rules]
        self._model_invariants = [
            (ModelView(self, owner, roles.get(owner)), owner, invariant) for owner, invariant in problem.model_invariants
        ]
        time = getattr(problem.system, 'time', None)
        self._phases = _phase_plan(time.phases if isinstance(time, DiscreteTime) else (), problem)
        owners = [(item.path, type(item.interaction)) for item in problem.interactions] + [
            (spec.path, spec.element.element if isinstance(spec.element, Group) else type(spec.element))
            for spec in problem.agents
        ]
        for path, component in owners:
            for name in getattr(component, 'reports', ()):
                self.reports_[f'{path}.{name}'] = math.nan
        for spec in problem.agents:
            self._populate(spec)
        self._populating = False
        self._check_invariants()

    def _agent_class(self, spec: CompiledAgents) -> type:
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
        stepper = self
        model = self._views[spec.path]

        def step(agent: mesa.Agent) -> None:
            """Run the rules of the agent's type unless the agent was removed earlier in the step."""
            view = AgentView(agent, stepper)
            for rule in spec.rules:
                if getattr(agent, 'removed_', False):
                    return
                rule.function(view, model)

        agent_class = type(f'{spec.path.replace("/", "_")}_agent', (base,), {'step': step})
        self._classes[agent_class] = spec
        return agent_class

    def _place(self, agent: mesa.Agent, location: object) -> None:
        """Place a new agent at a location, or at a random place that respects the space's capacity."""
        space = self.problem_.agent_space
        if isinstance(space, ContinuousSpace):
            point = location if location is not None else self.model_.rng.uniform([0, 0], [space.width, space.height])
            agent.position = np.asarray(point, dtype=float)
        elif space is not None:
            if location is not None:
                agent.cell = self.space_[_cell_key(location)]
            elif space.capacity is not None:
                agent.cell = self.space_.select_random_empty_cell()
            else:
                agent.cell = self.space_.all_cells.select_random_cell()

    def resolve_group(self, group: str, scope: str = '') -> str:
        """Return the path of a group named relative to the system of a component, or by its path.

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

    def create_agent(self, group: str, location: object, values: Mapping[str, object]) -> mesa.Agent:
        """Create an agent of a group with initial state values, placed at a location or a random place.

        Args:
            group:
                The path of the group.

            location:
                The location, or `None` for a random place.

            values:
                The initial values of states, with the element type's defaults for the others.

        Returns:
            agent:
                The Mesa agent.

        Raises:
            SimulationError:
                The group does not exist, or it is not variable, so its size cannot change.
        """
        spec = self._specs[self.resolve_group(group)]
        if not self._populating:
            _check_variable(spec, 'create')
        agent_class = self._agent_class(spec)
        agent = (
            agent_class(self.space_, self.model_)
            if agent_class.__mro__[1] is ContinuousSpaceAgent
            else agent_class(self.model_)
        )
        defaults = _agent_type(spec)
        for name in spec.states:
            value = values[name] if name in values else _shared_default(defaults, name)
            setattr(agent, name, value)
        self._place(agent, location)
        return agent

    def _populate(self, spec: CompiledAgents) -> None:
        """Create the initial agents of a group with their per-agent or shared initial values."""
        space = self.problem_.agent_space
        network_nodes = list(space.graph.nodes) if isinstance(space, NetworkSpace) else []
        given = spec.element.location if isinstance(spec.element, Group) else None
        for index in range(spec.size):
            values = {name: _member_value(value, index, spec.size) for name, value in spec.values.items()}
            location = _member_location(given, index, spec.size)
            if location is None and spec.size <= len(network_nodes):
                location = network_nodes[index]
            self.create_agent(spec.path, location, values)

    def spec_of(self, agent: mesa.Agent) -> CompiledAgents:
        """Return the group of a Mesa agent created by the stepper.

        Args:
            agent:
                The Mesa agent.

        Returns:
            spec:
                The compiled agent group.
        """
        return self._classes[type(agent)]

    def agents_of(self, group: str) -> list[Any] | None:
        """Return the alive Mesa agents of a group, or `None` when the model has no such group.

        Args:
            group:
                The path of the group.

        Returns:
            agents:
                The Mesa agents, or `None`.
        """
        if not any(spec.path == group for spec in self.problem_.agents):
            return None
        return self.group_agents(group)

    def group_agents(self, group: str) -> list[mesa.Agent]:
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
                return list(self.model_.agents_by_type.get(agent_class, []))
        return []

    def _push_parameters(self, p: Vector) -> None:
        """Make new parameter values visible to the rules."""
        self.parameters_ = p.copy()

    def _run_rules(self) -> None:
        """Run the phases of a step, or activate the agents in random order and run the model rules once."""
        if self._phases is None:
            self.model_.agents.shuffle_do('step')
            for model, rule in self._rules:
                rule.function(model)
        else:
            for phase in self._phases:
                for index in range(phase.repeat):
                    self.round_ = index
                    self._run_phase(phase)
            self.round_ = 0
        self._remember()
        self._check_invariants()

    def _check_invariants(self) -> None:
        """Check the invariants of every agent and interaction against their values when they were first checked."""
        for spec in self._specs.values():
            for agent in self.group_agents(spec.path) if spec.invariants else ():
                view = AgentView(agent, self)
                for invariant in spec.invariants:
                    key = (agent.unique_id, spec.path, invariant.label)
                    first = self._invariant_values.setdefault(key, invariant.value_of(view))
                    invariant.check(view, first, f'agent {view.id} of {spec.path}')
        for model, owner, invariant in self._model_invariants:
            first = self._invariant_values.setdefault((-1, owner, invariant.label), invariant.value_of(model))
            invariant.check(model, first, owner)

    def _ordered_agents(self, phase: Phase) -> list[mesa.Agent]:
        """Return the agents of a phase in the order of its activation policy."""
        if phase.activation == 'sequential':
            return sorted(self.model_.agents, key=lambda agent: agent.unique_id)
        if phase.activation == 'by-type':
            paths = [
                self.resolve_group(group) for group in phase.groups or [spec.path for spec in self.problem_.agents]
            ]
            ordered: list[mesa.Agent] = []
            for path in paths:
                members = sorted(self.group_agents(path), key=lambda agent: agent.unique_id)
                ordered += [members[index] for index in self.model_.rng.permutation(len(members))]
            return ordered
        agents = sorted(self.model_.agents, key=lambda agent: agent.unique_id)
        return [agents[index] for index in self.model_.rng.permutation(len(agents))]

    def _run_phase(self, phase: Phase) -> None:
        """Run the agent rules of a phase under its activation policy, then the model rules of the phase."""
        if phase.activation == 'simultaneous':
            self.buffer_ = {}
        try:
            for agent in self._ordered_agents(phase):
                spec = self.spec_of(agent)
                view, model = AgentView(agent, self), self._views[spec.path]
                for rule in spec.rules:
                    if getattr(agent, 'removed_', False):
                        break
                    if _phase_of(rule) == phase.name:
                        rule.function(view, model)
        finally:
            buffer, self.buffer_ = self.buffer_, None
        for (agent, name), value in (buffer or {}).items():
            setattr(agent, name, value)
        for model, rule in self._rules:
            if _phase_of(rule) == phase.name:
                rule.function(model)

    def _remember(self) -> None:
        """Record the states that the element types of agents remember, at the end of a step."""
        remembering = {}
        for spec in self.problem_.agents:
            agent_type = spec.element.element if isinstance(spec.element, Group) else type(spec.element)
            remembering[spec.path] = getattr(agent_type, 'remembers', None) or {}
        if not any(remembering.values()):
            return
        for agent in self.model_.agents:
            remembers = remembering[self.spec_of(agent).path]
            if not remembers:
                continue
            memories = agent.__dict__.setdefault('memories_', {})
            for name, window in remembers.items():
                memories.setdefault(name, deque(maxlen=window)).append(getattr(agent, name))

    def set_parameter(self, name: str, value: float) -> None:
        """Set a parameter during a step, so the rules read it at once and the trajectory records it.

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

    def observable(self, name: str) -> Any:  # noqa: ANN401
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
            self._evaluator = ObservableEvaluator(self.problem_)
        values = self._evaluator.evaluate(float(self.model_.time), self._state, self.parameters_, self.agents_of)
        if name not in values:
            error_msg = f'No observable named {name!r}. Available observables are: {", ".join(values) or "none"}.'
            raise SimulationError(error_msg)
        return values[name]

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
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
            self.model_.step()
        return x

    def snapshot(self) -> AgentState:
        """Return the agents of every group and the reporters at the current step.

        Returns:
            state:
                The agents and reporters.
        """
        groups = {}
        for spec in self.problem_.agents:
            agents = self.group_agents(spec.path)
            groups[spec.path] = _snapshot_of(agents, {name: _attribute(name) for name in spec.states})
        reporters = {f'{path}.count': float(len(snapshot)) for path, snapshot in groups.items()}
        return AgentState(groups, {**reporters, **self.reports_})


def _phase_of(rule: Rule) -> str:
    """Return the name of the phase a rule runs in."""
    return rule.phase or DEFAULT_PHASE


def _phase_plan(phases: Sequence[Phase], problem: Problem) -> list[Phase] | None:
    """Return the phases of a step with the default phase, or `None` when no phase is declared or named."""
    rules = [rule for spec in problem.agents for rule in spec.rules] + [rule for _, rule in problem.model_rules]
    if not phases and all(rule.phase is None for rule in rules):
        return None
    names = [phase.name for phase in phases]
    if len(set(names)) != len(names):
        error_msg = f'Phase names must be unique, and they are: {", ".join(names)}.'
        raise SimulationError(error_msg)
    plan = list(phases) if DEFAULT_PHASE in names else [Phase(DEFAULT_PHASE), *phases]
    unknown = sorted({_phase_of(rule) for rule in rules} - {phase.name for phase in plan})
    if unknown:
        error_msg = (
            f'Rules run in the phases {", ".join(unknown)}, which the system does not declare. Declared phases are: '
            f'{", ".join(names) or "none"}.'
        )
        raise SimulationError(error_msg)
    return plan


def _attribute(name: str) -> Callable[[mesa.Agent], object]:
    """Return a function that reads an attribute of an agent."""
    return lambda agent: getattr(agent, name)


def _snapshot_of(agents: list[mesa.Agent], reporters: Mapping[str, Callable[[mesa.Agent], object]]) -> AgentSnapshot:
    """Return the ids, reported values, and locations of agents."""
    agents = sorted(agents, key=lambda agent: agent.unique_id)
    ids = np.array([agent.unique_id for agent in agents], dtype=np.int64)
    values = {name: np.asarray([reporter(agent) for agent in agents]) for name, reporter in reporters.items()}
    locations = [_location_of(agent) for agent in agents]
    has_locations = any(location is not None for location in locations)
    return AgentSnapshot(ids, values, np.asarray(locations) if has_locations else None)


def _check_variable(spec: CompiledAgents, action: str) -> None:
    """Raise an error when a rule changes the size of an agent element or of a group that is not variable."""
    if isinstance(spec.element, Group) and spec.element.variable:
        return
    error_msg = (
        f'A rule tries to {action} an agent of {spec.path}, whose size is fixed. Declare the group with '
        '`variable=True`, such as `Element.group(label, size, variable=True)`, for its members to be created and '
        'removed during a run.'
    )
    raise SimulationError(error_msg)


def _member_value(value: object, index: int, size: int) -> object:
    """Return the initial value of one agent from a per-agent array or a shared value."""
    array = np.asarray(value)
    if array.ndim and array.shape[0] == size:
        item = array[index]
        return item.item() if np.ndim(item) == 0 else item
    return value


def _member_location(location: object, index: int, size: int) -> object:
    """Return the initial location of one agent from per-agent locations or one shared location."""
    if location is None:
        return None
    array = np.asarray(location, dtype=object if isinstance(location, list | tuple) else None)
    if array.ndim >= 1 and array.shape[0] == size:
        return _cell_key(np.asarray(array[index]))
    return _cell_key(location)


def _shared_default(element_type: type, name: str) -> object:
    """Return the default of a state of an element type."""
    field = next(field for field in dataclasses.fields(element_type) if field.name == name)
    factory = field.default_factory
    return factory() if callable(factory) else field.default


register_solver('mesa', MesaStepper, default=True)
