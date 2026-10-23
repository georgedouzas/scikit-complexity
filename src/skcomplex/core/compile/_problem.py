"""Define the compiled problem that solvers evaluate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from ..base import Expression, Form, SymbolicVector, VisualSpec, real_components_of
from ..components import Element, Group
from ..interactions import Interaction
from ..laws import Invariant, Rule
from ..space import Space
from ..system import System
from ._contractions import CompiledEncounter
from ._hybrid import Breakpoints, CompiledEvent, Delay
from ._lattices import LatticeStep, lattice_shape
from ._layout import Layout
from ._quantities import StockFlow
from ._reductions import CompiledReduction

ROW_TOLERANCE = 1e-9

type Vector = npt.NDArray[np.float64]
type VectorFunction = Callable[[float, Vector, Vector], Vector]


@dataclass(frozen=True)
class RenewalRecord:
    """Renewals of one renewal law that fired in one pass, each at the time of its member.

    Args:
        label:
            The name of the renewal law.

        times:
            The time of every renewal.

        members:
            The member of every renewal.

        before:
            The states of every renewing member before its reset, one row per renewal and one column per state.

        after:
            The states of every renewing member after its reset.

        states:
            The dotted state name, with the component of a vector state, of every column.
    """

    label: str
    times: Vector
    members: npt.NDArray[np.intp]
    before: Vector
    after: Vector
    states: tuple[str, ...]

    @property
    def time(self) -> float:
        """Time of the last renewal of the pass."""
        return float(np.max(self.times))


class RenewalProcess(Protocol):
    """Renewal law compiled for the members of its component, which renew on their own clocks."""

    @property
    def label(self) -> str:
        """Name of the renewal law."""

    @property
    def members(self) -> int:
        """Number of members that renew on their own clocks."""

    @property
    def counts(self) -> npt.NDArray[np.intp]:
        """Flat index of the hidden state that counts the renewals of every member."""

    @property
    def states(self) -> npt.NDArray[np.intp]:
        """Flat indices of the states of every member, one row per member."""

    def start(self, t: float, x: Vector, p: Vector, rng: np.random.Generator) -> Vector:
        """Return the first renewal time of every member from the states at a start time."""

    def fire(
        self,
        times: Vector,
        fired: npt.NDArray[np.intp],
        x: Vector,
        p: Vector,
        rng: np.random.Generator,
    ) -> tuple[Vector, Vector, RenewalRecord]:
        """Renew members at their times, and return the state, their next renewal times, and the record."""


@dataclass(frozen=True)
class NetworkSnapshot:
    """Nodes and edges of a network layer at one frame.

    Args:
        nodes:
            The rows of the nodes present, in increasing order.

        sources:
            The node row of the source of every edge present.

        targets:
            The node row of the target of every edge present.

        edges:
            The row of every edge present in its edge group.
    """

    nodes: npt.NDArray[np.intp]
    sources: npt.NDArray[np.intp]
    targets: npt.NDArray[np.intp]
    edges: npt.NDArray[np.intp]


class TopologyChanges(Protocol):
    """Topology requests of a problem, which the stepper applies between steps, and the networks they change."""

    @property
    def channels(self) -> Mapping[int, object]:
        """Requests of jump laws by the index of their jump channels."""

    @property
    def layers(self) -> tuple[tuple[str, str | None], ...]:
        """Path of the first group on the nodes and the layer of every network layer with a variable edge group."""

    def decide(self, t: float, x: Vector, p: Vector, rng: np.random.Generator) -> list[tuple[int, int]]:
        """Return the requests of update laws that fire at a state, each as its index and the row it fires for."""

    def apply(
        self,
        fired: list[tuple[int, int]],
        t: float,
        x: Vector,
        p: Vector,
        rng: np.random.Generator,
    ) -> Vector:
        """Apply fired requests of update laws to a state and to the parameters in place, and return the state."""

    def fire(self, channel: int, row: int, t: float, x: Vector, p: Vector, *, rng: np.random.Generator) -> Vector:
        """Apply the request of a jump channel for a row, and return the state."""

    def read_network(self, p: Vector, nodes: str | None = None, layer: str | None = None) -> NetworkSnapshot:
        """Return the nodes and edges of a layer of a network at the parameters of a frame."""


@dataclass(frozen=True)
class FractionalRows:
    """Noise rows of one fractional noise law, driven by fractional Brownian motion of a Hurst index.

    Args:
        law:
            The description of the law, such as `fractional noise law of walkers.x`.

        positions:
            The positions of the law's rows among the noise rows of the problem.

        hurst:
            The function `H(p)` that returns the Hurst index of every row from the parameter vector.
    """

    law: str
    positions: npt.NDArray[np.intp]
    hurst: Callable[[Vector], Vector]


class Jacobian(Protocol):
    """Exact Jacobian of a drift with its sparsity pattern."""

    def __call__(self, t: float, x: Vector, p: Vector) -> Any:  # noqa: ANN401
        """Return the Jacobian at a state as a sparse matrix."""

    def pattern(self) -> Any:  # noqa: ANN401
        """Return the boolean structure of the Jacobian, or `None` when it is too dense to state."""

    def validate(self) -> None:
        """Raise an error when a law has no symbolic form whose exact Jacobian can be derived."""


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
    """Agent element or agent group of an agent-based problem, whose states live in the Mesa model.

    Args:
        path:
            The slash-separated path of the agents in the system tree.

        element:
            The agent element or the agent group.

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

        invariants:
            The invariants of every agent, functions of the agent the stepper checks after every step.
    """

    path: str
    element: Element | Group
    size: int
    states: tuple[str, ...]
    values: Mapping[str, object]
    parameters: Mapping[str, str]
    rules: tuple[Rule, ...]
    visual: VisualSpec
    invariants: tuple[Invariant, ...] = ()


@dataclass(frozen=True)
class CompiledInteraction:
    """Interaction of a compiled problem with the paths its roles are bound to.

    Args:
        path:
            The slash-separated path of the interaction in the system tree.

        interaction:
            The interaction.

        roles:
            A mapping from role name to the path of the element, group, or observable bound to it, or to the paths
            of the elements of an element set.
    """

    path: str
    interaction: Interaction
    roles: Mapping[str, str | tuple[str, ...]]


@dataclass(frozen=True)
class JumpChannels:
    """Random events of one jump law, one channel per evaluation index.

    Args:
        offset:
            The index of the first channel among all channels of the problem.

        targets:
            The flat state index changed by each change, one row per change and one column per channel, or -1 where
            a change at a member offset falls past a fixed end and changes nothing.

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
class WienerBlock:
    """Correlated Wiener processes of one Wiener element, a contiguous block of the processes of a problem.

    Args:
        start:
            The index of the first process of the block.

        size:
            The number of processes of the block.

        factor:
            The factor `L` with `L L^T = C` of the covariance `C` of the block.
    """

    start: int
    size: int
    factor: npt.NDArray[np.float64]


@dataclass(frozen=True)
class SampledDraw:
    """Draw from a `scipy.stats` distribution that a stochastic map takes for every member at every step.

    Args:
        start:
            The index of the first value in the extended parameter vector.

        rows:
            The number of values, one per member.

        sample:
            The function `sample(t, x, q, rng)` that returns the values, given the extended parameter vector `q`.
    """

    start: int
    rows: int
    sample: Callable[[float, Vector, Vector, np.random.Generator], Vector]


@dataclass(frozen=True)
class Draws:
    """Random values a stochastic map draws at every step, stored after the parameters of an extended vector.

    Args:
        offset:
            The index of the first draw in the extended vector, the size of the parameter vector.

        normal:
            One flag per uniform or normal draw, true for a standard normal value and false for a uniform value in
            `[0, 1)`, in the order the draws are taken, by component path, draw name, and member index.

        sampled:
            The draws from `scipy.stats` distributions, stored after the uniform and normal values and taken after
            them.
    """

    offset: int
    normal: npt.NDArray[np.bool_]
    sampled: tuple[SampledDraw, ...] = ()

    @property
    def size(self) -> int:
        """Number of values drawn at every step."""
        return int(self.normal.size) + sum(item.rows for item in self.sampled)

    def extend(self, p: Vector) -> Vector:
        """Return a parameter vector extended with room for the draws.

        Args:
            p:
                The parameter vector.

        Returns:
            extended:
                A new vector with the parameters first and zeros for the draws.
        """
        extended = np.zeros(self.offset + self.size)
        extended[: self.offset] = p[: self.offset]
        return extended

    def fill(self, q: Vector, rng: np.random.Generator, t: float = 0.0, x: Vector | None = None) -> None:
        """Draw fresh values into an extended parameter vector, in order, uniform and normal runs as they come.

        Args:
            q:
                The extended parameter vector, changed in place.

            rng:
                The random number generator.

            t:
                The time, which the arguments of draws from `scipy.stats` distributions may read.

            x:
                The state vector, which the arguments of draws from `scipy.stats` distributions may read.
        """
        cuts = [0, *(np.flatnonzero(np.diff(self.normal)) + 1).tolist(), self.normal.size]
        for start, end in itertools.pairwise(cuts):
            if start == end:
                continue
            count = end - start
            values = rng.standard_normal(count) if self.normal[start] else rng.random(count)
            q[self.offset + start : self.offset + end] = values
        state = np.zeros(0) if x is None else x
        for item in self.sampled:
            q[item.start : item.start + item.rows] = item.sample(t, state, q, rng)


@dataclass(frozen=True)
class MarkovMatrix:
    """Transition or emission matrix of a law, evaluated from the parameter vector to check its rows.

    Args:
        law:
            The description of the law, such as `transition law of chain.x`.

        function:
            The function `f(p)` that returns the matrix of every evaluation row, of shape `(rows, n, m)`.

        reads:
            The indices of the parameter vector that the matrix reads.
    """

    law: str
    function: Callable[[Vector], npt.NDArray[np.float64]]
    reads: frozenset[int]

    def violation(self, p: Vector) -> str | None:
        """Return the description of the first negative entry or row that does not sum to one, or `None`.

        Args:
            p:
                The parameter vector.

        Returns:
            violation:
                The description of what is wrong with the matrix, or `None` when it is a stochastic matrix.
        """
        values = self.function(p)
        for member, matrix in enumerate(values):
            where = f' of member {member}' if values.shape[0] > 1 else ''
            negative = np.argwhere(~(matrix >= 0))
            if negative.size:
                row, column = negative[0]
                return f'the entry ({row}, {column}){where} of the {self.law} is {matrix[row, column]}, below 0'
            sums = matrix.sum(axis=1)
            wrong = np.flatnonzero(np.abs(sums - 1) > ROW_TOLERANCE)
            if wrong.size:
                return f'the row {wrong[0]}{where} of the {self.law} sums to {sums[wrong[0]]}, not 1'
        return None


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
            The discrete map loop `iterate(t, x, p, until)`, or `iterate(t, x, p, until, rng)` for a stochastic map.
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
            The agent elements and groups of an agent-based problem.

        model_rules:
            The rules of interactions of an agent-based problem, with the path of their interaction.

        model_invariants:
            The invariants of interactions of an agent-based problem, functions of the interaction's model view, with
            the path of their interaction.

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

        lattice:
            The local rules, drives, relaxations, and Monte Carlo sampling of a lattice problem, in the order they
            apply within a step.

        linear:
            The linear terms with constant coefficients of the rates of fields, which implicit and exponential solvers
            treat exactly while the rest of the rates is the nonlinear remainder.

        volumes:
            The conservation laws of finite-volume fields, whose flux differences the drift includes.

        constrained:
            The state indices held at their initial values, the fixed boundary nodes of finite element fields.

        reductions:
            The reductions that laws read, in the order they are evaluated before the kernels.

        jacobian:
            The Jacobian of the drift of an ODE problem, derived from its laws: its sparsity pattern and, for laws with
            a symbolic form, its exact value. `None` for other forms.

        stable_step:
            The largest step of explicit fixed-step solvers for which the diffusion of fields is stable,
            `h^2 / (degree * D)`, or `None` when the system has no diffusion with numeric coefficients.

        diffusive_limits:
            The stability limits of the diffusion of finite-volume fields, which depend on the parameters.

        drift:
            The function `f(t, x, p)` that returns the time derivative or the drift.

        diffusion:
            The function `g(t, x, p)` that returns every noise amplitude, one per state and process it couples.

        noise_rows:
            The state index each amplitude of `diffusion` drives.

        noise_columns:
            The Wiener process each amplitude of `diffusion` draws, which several amplitudes share for shared noise.

        n_processes:
            The number of standard normal increments drawn per step.

        noise_blocks:
            The blocks of correlated processes, whose increments their covariance factor scales.

        propensities:
            The function `a(t, x, p)` that returns the rate of every jump channel.

        jumps:
            The channels of every jump law.

        update:
            The function `F(t, x, p)` that returns the next state vector, or `F(t, x, p, rng)` for a stochastic map.

        boundary:
            The function `B(t, x, p)` that returns the state after the boundary laws.

        draws:
            The random values the update draws at every step, or `None` for a deterministic map.

        matrices:
            The transition and emission matrices of the Markov laws, checked when their parameters change.

        renewals:
            The renewal laws, whose members renew on their own clocks.

        fractional:
            The noise rows that fractional noise laws drive, with their Hurst indices.

        orders:
            The order of the Caputo derivative that the drift gives for every state component of a problem of the
            fractional form, 1 for the ordinary states, and empty for the other forms.

        stock_flow:
            The quantities, flows, auxiliaries, and feedback terms of the system, or `None` for partitioned and Mesa
            problems.

        encounters:
            The encounter laws, whose gain and loss rates the drift holds and particle solvers sample.

        backend:
            The name of the backend that produced the numeric functions.

        accelerated:
            The compiled stepper loops, or `None` when the backend has none for this problem.

        rewriting:
            The compiled rewriting systems of the words, patches, and host graphs of a rewriting system.

        topology:
            The topology requests of the laws and the networks they change, which the stepper applies between steps,
            or `None` for a problem without them.
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
    model_invariants: tuple[tuple[str, Invariant], ...] = ()
    agent_space: Space | None = None
    partitions: tuple['Partition', ...] = ()
    companions: Mapping[str, tuple[str, str, int]] = field(default_factory=dict)
    delays: tuple[Delay, ...] = ()
    events: tuple[CompiledEvent, ...] = ()
    breakpoints: Breakpoints = field(default_factory=Breakpoints)
    history: Mapping[str, Any] = field(default_factory=dict)
    lattice: tuple[LatticeStep, ...] = field(default=(), repr=False)
    stable_step: float | None = None
    diffusive_limits: tuple[Any, ...] = field(default=(), repr=False)
    linear: tuple[Any, ...] = field(default=(), repr=False)
    volumes: tuple[Any, ...] = field(default=(), repr=False)
    constrained: npt.NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp), repr=False)
    reductions: tuple[CompiledReduction, ...] = field(default=(), repr=False)
    jacobian: Jacobian | None = field(default=None, repr=False)
    drift: VectorFunction | None = field(default=None, repr=False)
    diffusion: VectorFunction | None = field(default=None, repr=False)
    noise_rows: npt.NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp), repr=False)
    noise_columns: npt.NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp), repr=False)
    n_processes: int = 0
    noise_blocks: tuple[WienerBlock, ...] = field(default=(), repr=False)
    propensities: VectorFunction | None = field(default=None, repr=False)
    jumps: tuple[JumpChannels, ...] = field(default=(), repr=False)
    update: Callable[..., Vector] | None = field(default=None, repr=False)
    boundary: VectorFunction | None = field(default=None, repr=False)
    draws: Draws | None = field(default=None, repr=False)
    matrices: tuple[MarkovMatrix, ...] = field(default=(), repr=False)
    renewals: tuple[RenewalProcess, ...] = field(default=(), repr=False)
    fractional: tuple[FractionalRows, ...] = field(default=(), repr=False)
    orders: Vector = field(default_factory=lambda: np.zeros(0), repr=False)

    stock_flow: StockFlow | None = field(default=None, repr=False)
    encounters: tuple[CompiledEncounter, ...] = field(default=(), repr=False)
    backend: str = 'numpy'
    accelerated: Accelerated | None = field(default=None, repr=False)
    rewriting: tuple[Any, ...] = field(default=(), repr=False)
    topology: TopologyChanges | None = field(default=None, repr=False)

    def _flatten_symbols(self, layout: Layout) -> list[Expression]:
        """Return the symbols of the entries of a layout as a flat list."""
        return [item for entry in layout for item in real_components_of(self.symbols[entry.name])]

    @property
    def stochastic(self) -> bool:
        """Whether the update draws random values, so it takes the random number generator."""
        return self.draws is not None

    @property
    def state_symbols(self) -> list[Expression]:
        """Symbols of the state components of single elements and of one member of every group."""
        return self._flatten_symbols(self.states)

    @property
    def parameter_symbols(self) -> list[Expression]:
        """Symbols of the parameter components of interactions, single elements, and one member of every group."""
        return self._flatten_symbols(self.parameters)

    def max_speed(self, x: Vector, p: Vector) -> float | None:
        """Return the largest wave speed of the conservation laws, or `None` without them.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            speed:
                The largest wave speed over the cells, the dimensions, and the finite-volume fields.
        """
        if not self.volumes:
            return None
        return max(volume.max_speed(x, p) for volume in self.volumes)

    def diffusive_step(self, p: Vector) -> float | None:
        """Return the largest stable explicit step for the diffusion of finite-volume fields, or `None` without it.

        Args:
            p:
                The parameter vector.

        Returns:
            step:
                The smallest limit over the finite-volume fields at the parameter values.
        """
        if not self.diffusive_limits:
            return None
        return min(limit.step(p) for limit in self.diffusive_limits)

    def lattice_shape(self, name: str) -> tuple[int, ...] | None:
        """Return the lattice shape of a state that lattice laws read or change, or `None` for other states.

        Args:
            name:
                The dotted name of the state.

        Returns:
            shape:
                The number of sites along every dimension.
        """
        found = lattice_shape(name, self.lattice)
        if found is not None:
            return found
        owner = name.rpartition('.')[0]
        for element in self.elements:
            space = getattr(element.element, 'space', None)
            if element.path == owner and hasattr(space, 'dimensions'):
                return space.dimensions  # type: ignore[union-attr]
        return None

    def draw_increments(self, rng: np.random.Generator, scale: float) -> Vector:
        """Return the Wiener increment that every noise amplitude multiplies, drawn for one step.

        The increments of the processes are normal with the standard deviation `scale`, correlated within every block
        by its covariance factor. Amplitudes that share a process take its increment.

        Args:
            rng:
                The random number generator of the run.

            scale:
                The square root of the step.

        Returns:
            increments:
                One increment per noise amplitude, in the order of `noise_rows`.
        """
        increments = rng.normal(0.0, scale, size=self.n_processes)
        for block in self.noise_blocks:
            part = slice(block.start, block.start + block.size)
            increments[part] = block.factor @ increments[part]
        return increments[self.noise_columns]

    @property
    def stoichiometry(self) -> Vector:
        """Increments of the state vector, one row per jump channel."""
        n_channels = sum(channels.size for channels in self.jumps)
        matrix = np.zeros((n_channels, self.states.size))
        for channels in self.jumps:
            rows = channels.offset + np.arange(channels.size)
            for targets, increment in zip(channels.targets, channels.increments, strict=True):
                kept = targets >= 0
                np.add.at(matrix, (rows[kept], targets[kept]), increment)
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
