"""Define the compiled problem that solvers evaluate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from ..core import Expression, Form, SymbolicVector, VisualSpec, components_of
from ..model import Element, Group, Interaction, Rule, Space, System
from ._hybrid import Breakpoints, CompiledEvent, Delay
from ._layout import Layout

type Vector = npt.NDArray[np.float64]
type VectorFunction = Callable[[float, Vector, Vector], Vector]


@dataclass(frozen=True)
class CompiledElement:
    """Element or group of a compiled problem with the dotted names of its variables.

    Args:
        path:
            The slash-separated path of the element or group in the system tree.

        element:
            The element or group.

        states:
            A mapping from state name to dotted name.

        parameters:
            A mapping from parameter name to dotted name.

        visual:
            The visual that draws the element or group.

        members:
            The size of a group, or `None` for a single element.
    """

    path: str
    element: Element | Group
    states: Mapping[str, str]
    parameters: Mapping[str, str]
    visual: VisualSpec
    members: int | None = None


@dataclass(frozen=True)
class CompiledAgents:
    """Agent type or agent group of an agent-based problem, whose states live in the Mesa model.

    Args:
        path:
            The slash-separated path of the agents in the system tree.

        element:
            The agent type instance or the agent group.

        size:
            The initial number of agents.

        states:
            The names of the states every agent holds.

        values:
            A mapping from state name to its initial per-agent array or shared value.

        parameters:
            A mapping from parameter name to dotted name in the parameter vector.

        rules:
            The rules every agent runs each step, in declaration order.

        visual:
            The visual that draws the agents.
    """

    path: str
    element: Element | Group
    size: int
    states: tuple[str, ...]
    values: Mapping[str, object]
    parameters: Mapping[str, str]
    rules: tuple[Rule, ...]
    visual: VisualSpec


@dataclass(frozen=True)
class CompiledInteraction:
    """Interaction of a compiled problem with the paths its roles are bound to.

    Args:
        path:
            The slash-separated path of the interaction in the system tree.

        interaction:
            The interaction.

        roles:
            A mapping from role name to the path of the element or group bound to it.
    """

    path: str
    interaction: Interaction
    roles: Mapping[str, str]


@dataclass(frozen=True)
class JumpChannels:
    """Random events of one jump law, one channel per evaluation index.

    Args:
        offset:
            The index of the first channel among all channels of the problem.

        targets:
            The flat state index changed by each change, one row per change and one column per channel.

        increments:
            The increment of each change.
    """

    offset: int
    targets: npt.NDArray[np.intp]
    increments: Vector

    @property
    def size(self) -> int:
        """Number of channels."""
        return int(self.targets.shape[1])


@dataclass(frozen=True)
class Accelerated:
    """Compiled stepper loops of a problem, which run every internal step of a frame in compiled code.

    Every loop takes `(t, x, p, until, ...)` and returns the state at `until`. A loop is `None` when the problem's form
    does not use it.

    Args:
        rk4:
            The fourth-order Runge-Kutta loop `rk4(t, x, p, until, h)`.

        velocity_verlet:
            The velocity Verlet loop `velocity_verlet(t, x, p, until, h, positions, velocities)`.

        euler_maruyama:
            The Euler-Maruyama loop `euler_maruyama(t, x, p, until, h, rng)`.

        gillespie:
            The Gillespie loop `gillespie(t, x, p, until, rng)`.

        iterate:
            The discrete map loop `iterate(t, x, p, until)`.
    """

    rk4: Callable[..., Vector] | None = None
    velocity_verlet: Callable[..., Vector] | None = None
    euler_maruyama: Callable[..., Vector] | None = None
    gillespie: Callable[..., Vector] | None = None
    iterate: Callable[..., Vector] | None = None


@dataclass(frozen=True)
class Problem:
    """System compiled to one mathematical form with flat state and parameter vectors.

    Only the functions of the problem's form are set. `drift`, `diffusion`, and `noise_rows` belong to the ODE and SDE
    forms, `propensities` and `jumps` to the jump process form, and `update` to the discrete map form. `boundary`
    belongs to any form whose system has boundary laws.

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

        equations:
            A mapping from state component name to its symbolic drift, or to its next value for the discrete form.
            Group components are named with the member index `i`. Components with a callable law, a sum over group
            members, or a sum over pairs are left out.

        equation_symbols:
            A mapping from the names of `equations` to the symbol of the component.

        has_callables:
            Whether any law is a callable.

        has_group_sums:
            Whether any law sums over group members or pairs.

        elements:
            The compiled elements and groups in state vector order.

        interactions:
            The compiled interactions in tree order.

        system:
            The system the problem was compiled from.

        agents:
            The agent types and groups of an agent-based problem.

        model_rules:
            The rules of interactions of an agent-based problem, with the path of their interaction.

        agent_space:
            The space the agents of an agent-based problem live in, or `None`.

        companions:
            A mapping from the name of every companion state to its kind, `'derivative'` or `'lag'`, the name of the
            state it belongs to, and the order of the derivative or the number of steps back.

        partitions:
            The partitions of a co-simulation problem, in system order.

        delays:
            The delayed states that laws read, which solvers fill from the run's history.

        events:
            The discrete events, with their guard and reset functions.

        breakpoints:
            The times where time-dependent inputs jump, at which solvers restart.

        history:
            The values of states before the start, by dotted state path, as numbers or functions of time.

        drift:
            The function `f(t, x, p)` that returns the time derivative or the drift.

        diffusion:
            The function `g(t, x, p)` that returns the amplitude of every independent Wiener process.

        noise_rows:
            The state index each Wiener process drives.

        propensities:
            The function `a(t, x, p)` that returns the rate of every jump channel.

        jumps:
            The channels of every jump law.

        update:
            The function `F(t, x, p)` that returns the next state vector.

        boundary:
            The function `B(t, x, p)` that returns the state after the boundary laws.

        backend:
            The name of the backend that produced the numeric functions.

        accelerated:
            The compiled stepper loops, or `None` when the backend has none for this problem.
    """

    form: Form
    states: Layout
    parameters: Layout
    initial_state: Vector
    initial_parameters: Vector
    time: Expression
    symbols: Mapping[str, Expression | SymbolicVector]
    equations: Mapping[str, Expression]
    equation_symbols: Mapping[str, Expression]
    has_callables: bool
    has_group_sums: bool
    elements: tuple[CompiledElement, ...]
    interactions: tuple[CompiledInteraction, ...] = ()
    system: System | None = field(default=None, repr=False)
    agents: tuple[CompiledAgents, ...] = ()
    model_rules: tuple[tuple[str, Rule], ...] = ()
    agent_space: Space | None = None
    partitions: tuple['Partition', ...] = ()
    companions: Mapping[str, tuple[str, str, int]] = field(default_factory=dict)
    delays: tuple[Delay, ...] = ()
    events: tuple[CompiledEvent, ...] = ()
    breakpoints: Breakpoints = field(default_factory=Breakpoints)
    history: Mapping[str, Any] = field(default_factory=dict)
    drift: VectorFunction | None = field(default=None, repr=False)
    diffusion: VectorFunction | None = field(default=None, repr=False)
    noise_rows: npt.NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp), repr=False)
    propensities: VectorFunction | None = field(default=None, repr=False)
    jumps: tuple[JumpChannels, ...] = field(default=(), repr=False)
    update: VectorFunction | None = field(default=None, repr=False)
    boundary: VectorFunction | None = field(default=None, repr=False)
    backend: str = 'numpy'
    accelerated: Accelerated | None = field(default=None, repr=False)

    def _flatten_symbols(self, layout: Layout) -> list[Expression]:
        """Return the symbols of the entries of a layout as a flat list."""
        return [item for entry in layout for item in components_of(self.symbols[entry.name])]

    @property
    def state_symbols(self) -> list[Expression]:
        """Symbols of the state components of single elements and of one member of every group."""
        return self._flatten_symbols(self.states)

    @property
    def parameter_symbols(self) -> list[Expression]:
        """Symbols of the parameter components of interactions, single elements, and one member of every group."""
        return self._flatten_symbols(self.parameters)

    @property
    def stoichiometry(self) -> Vector:
        """Increments of the state vector, one row per jump channel."""
        n_channels = sum(channels.size for channels in self.jumps)
        matrix = np.zeros((n_channels, self.states.size))
        for channels in self.jumps:
            rows = channels.offset + np.arange(channels.size)
            for targets, increment in zip(channels.targets, channels.increments, strict=True):
                np.add.at(matrix, (rows, targets), increment)
        return matrix


@dataclass(frozen=True)
class Partition:
    """Part of a co-simulation problem, compiled with the laws of its own elements.

    The partition's problem has the state layout of the whole system. The states of other partitions have no laws in
    it, so they hold their values while it advances and act as its inputs.

    Args:
        name:
            The name of the partition, the path of its subsystem.

        problem:
            The problem of the partition.

        owned:
            The flat state indices of the partition's own elements.

        owners:
            The paths of the partition's own elements and groups.
    """

    name: str
    problem: Problem
    owned: npt.NDArray[np.intp]
    owners: frozenset[str]
