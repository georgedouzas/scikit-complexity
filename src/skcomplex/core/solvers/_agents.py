"""Run agent-based problems and wrapped Mesa models on Mesa, one Mesa step per unit of time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import mesa
import networkx as nx
import numpy as np
import numpy.typing as npt
from mesa.discrete_space import CellAgent, Network, OrthogonalMooreGrid, OrthogonalVonNeumannGrid
from mesa.experimental.continuous_space import ContinuousSpace as MesaContinuousSpace
from mesa.experimental.continuous_space import ContinuousSpaceAgent

from ..base import Form, SimulationError, SpaceError
from ..compile import CompiledAgents, Problem, Vector
from ..model import ContinuousSpace, GridSpace, Group, MesaSystem, NetworkSpace, Space
from ._base import Stepper, register_solver

MAX_SEED = 2**31 - 1


@dataclass(frozen=True)
class AgentSnapshot:
    """Agents of one group alive at one frame.

    Args:
        ids:
            The unique id of every agent.

        values:
            A mapping from state name to one value per agent.

        locations:
            One row per agent: grid coordinates, a network node, or a continuous position, or `None` without a space.
    """

    ids: npt.NDArray[np.int64]
    values: Mapping[str, npt.NDArray[Any]]
    locations: npt.NDArray[Any] | None = None

    def __len__(self) -> int:
        """Return the number of agents."""
        return len(self.ids)


@dataclass(frozen=True)
class AgentState:
    """Agents of every group and the model reporters at one frame.

    Args:
        groups:
            A mapping from group path to the agents of that group.

        reporters:
            A mapping from dotted reporter name to its value.
    """

    groups: Mapping[str, AgentSnapshot] = field(default_factory=dict)
    reporters: Mapping[str, float] = field(default_factory=dict)


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
                The initial values of states, with the agent type's defaults for the others.

        Returns:
            agent:
                The new agent.
        """
        return AgentView(self._stepper.create_agent(group, location, values), self._stepper)


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
        """Set a state of the agent."""
        if name not in self._stepper.spec_of(self._agent).states:
            error_msg = f'{name!r} is not a state of the agent, so it cannot be set.'
            raise AttributeError(error_msg)
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
        """Remove the agent from the model."""
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
                The initial values of states, with the agent type's defaults for the others.

        Returns:
            agent:
                The new agent.
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
    if isinstance(space, GridSpace):
        grid = OrthogonalVonNeumannGrid if space.neighborhood == 'von_neumann' else OrthogonalMooreGrid
        return grid((space.width, space.height), torus=space.torus, capacity=space.capacity, random=model.random)
    if isinstance(space, NetworkSpace):
        return Network(space.graph, capacity=space.capacity, random=model.random, layout=None)
    if isinstance(space, ContinuousSpace):
        bounds = [[0.0, space.width], [0.0, space.height]]
        return MesaContinuousSpace(bounds, torus=space.torus, random=model.random, n_agents=max(n_agents, 1))
    return None


class MesaStepper(Stepper):
    """Stepper of agent-based problems and wrapped Mesa models, one Mesa step per unit of time.

    Every step advances Mesa's clock by one, activates all agents in random order, and each agent runs the rules of its
    type in declaration order. Interaction rules run once after the agents. A wrapped Mesa model runs its own `step`.

    Attributes:
        model_:
            The Mesa model.

    Examples:
        >>> from skcomplex.core.base import state
        >>> from skcomplex.core.model import AgentType, Rule, System
        >>> from skcomplex.core.runtime import Simulation
        >>> class Counter(AgentType):
        ...     count: float = state(0.0)
        ...
        ...     def laws(self, s):
        ...         return [Rule(self.tick)]
        ...
        ...     def tick(self, agent, model):
        ...         agent.count += 1
        >>> trajectory = Simulation(System(elements=[Counter.group('counters', 3)])).run(until=4).trajectory_
        >>> trajectory.agents('counters', 4).values['count'].tolist()
        [4.0, 4.0, 4.0]
    """

    form: ClassVar[Form] = Form.AGENT

    def _seed(self) -> int:
        """Return a seed for Mesa drawn from the stepper's random number generator."""
        return int(self.rng_.integers(0, MAX_SEED))

    def initialize(self, problem: Problem, rng: np.random.Generator) -> None:
        """Build the Mesa model of an agent-based problem or of a wrapped Mesa model.

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
        if isinstance(problem.system, MesaSystem):
            self._build_wrapped(problem.system)
            return
        self._specs = {spec.path: spec for spec in problem.agents}
        self._classes: dict[type, CompiledAgents] = {}
        self.model_ = _AgentModel(self._run_rules, self._seed())
        self.space_: Any = _build_space(problem.agent_space, self.model_, sum(spec.size for spec in problem.agents))
        roles = {item.path: item.roles for item in problem.interactions}
        self._views = {spec.path: ModelView(self, spec.path) for spec in problem.agents}
        self._rules = [(ModelView(self, owner, roles.get(owner)), rule) for owner, rule in problem.model_rules]
        for spec in problem.agents:
            self._populate(spec)

    def _agent_class(self, spec: CompiledAgents) -> type:
        """Return the Mesa agent class of a group, created on first use."""
        for agent_class, known in self._classes.items():
            if known is spec:
                return agent_class
        space = self.problem_.agent_space
        base = (
            CellAgent if isinstance(space, GridSpace | NetworkSpace) else ContinuousSpaceAgent if space else mesa.Agent
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
                The initial values of states, with the agent type's defaults for the others.

        Returns:
            agent:
                The Mesa agent.

        Raises:
            SimulationError:
                The group does not exist.
        """
        spec = self._specs[self.resolve_group(group)]
        agent_class = self._agent_class(spec)
        agent = (
            agent_class(self.space_, self.model_)
            if agent_class.__mro__[1] is ContinuousSpaceAgent
            else agent_class(self.model_)
        )
        defaults = spec.element.element if isinstance(spec.element, Group) else type(spec.element)
        for name in spec.states:
            value = values[name] if name in values else _shared_default(defaults, name)
            setattr(agent, name, value)
        self._place(agent, location)
        return agent

    def _populate(self, spec: CompiledAgents) -> None:
        """Create the initial agents of a group with their per-agent or shared initial values."""
        space = self.problem_.agent_space
        network_nodes = list(space.graph.nodes) if isinstance(space, NetworkSpace) else []
        given = spec.element.values.get('location') if hasattr(spec.element, 'values') else spec.element.location  # type: ignore[union-attr]
        for index in range(spec.size):
            values = {name: _member_value(value, index, spec.size) for name, value in spec.values.items()}
            location = _member_location(given, index, spec.size)
            if location is None and spec.size <= len(network_nodes):
                location = network_nodes[index]
            self.create_agent(spec.path, location, values)

    def _build_wrapped(self, system: MesaSystem) -> None:
        """Build a wrapped Mesa model from its constructor arguments and a seed."""
        arguments = dict(system.parameters or {})
        for name, value in arguments.items():
            entry = f'{system.label}.{name}'
            if entry in self.problem_.parameters:
                arguments[name] = _typed_like(value, self.problem_.parameters.read(self.parameters_, entry))
        accepted = inspect.signature(system.model).parameters
        seed_name = 'rng' if 'rng' in accepted else 'seed'
        self.model_ = system.model(**arguments, **{seed_name: self._seed()})

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
        """Make new parameter values visible to the rules or set them on a wrapped model."""
        self.parameters_ = p.copy()
        system = self.problem_.system
        if isinstance(system, MesaSystem):
            for name in system.parameters or {}:
                entry = f'{system.label}.{name}'
                if entry in self.problem_.parameters:
                    value = self.problem_.parameters.read(p, entry)
                    setattr(self.model_, name, _typed_like(getattr(self.model_, name, value), value))

    def _run_rules(self) -> None:
        """Activate the agents in random order and run the model rules once."""
        self.model_.agents.shuffle_do('step')
        for model, rule in self._rules:
            rule.function(model)

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
        system = self.problem_.system
        if isinstance(system, MesaSystem):
            return _wrapped_snapshot(system, self.model_)
        groups = {}
        for spec in self.problem_.agents:
            agents = self.group_agents(spec.path)
            groups[spec.path] = _snapshot_of(agents, {name: _attribute(name) for name in spec.states})
        reporters = {f'{path}.count': float(len(snapshot)) for path, snapshot in groups.items()}
        return AgentState(groups, reporters)


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


def _wrapped_snapshot(system: MesaSystem, model: mesa.Model) -> AgentState:
    """Return the agents and reporters of a wrapped Mesa model."""
    reporters = {
        name: (lambda agent, attribute=reporter: getattr(agent, attribute)) if isinstance(reporter, str) else reporter
        for name, reporter in (system.agent_reporters or {}).items()
    }
    path = f'{system.label}/{system.agents}'
    snapshot = _snapshot_of(list(model.agents), reporters)
    values = {
        f'{system.label}.{name}': float(reporter(model)) for name, reporter in (system.model_reporters or {}).items()
    }
    values[f'{path}.count'] = float(len(snapshot))
    return AgentState({path: snapshot}, values)


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
    """Return the default of a state of an agent type."""
    field = next(field for field in dataclasses.fields(element_type) if field.name == name)
    factory = field.default_factory
    return factory() if callable(factory) else field.default


def _typed_like(reference: object, value: object) -> object:
    """Return a value with the integer type of a reference value when the reference is an integer."""
    number = float(np.ravel(np.asarray(value, dtype=float))[0])
    return round(number) if isinstance(reference, int) and not isinstance(reference, bool) else number


register_solver('mesa', MesaStepper, default=True)
