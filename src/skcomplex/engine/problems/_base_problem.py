"""Base of the problems a system compiles to: what every solver, observer, and visual reads of one."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, Self

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from ...core.errors import ValidationError
from ...core.spaces import BaseSpace
from ...core.specs import ObservableSpec
from ...core.system import System
from ...core.types import Vector
from ._compiled_agents import CompiledAgents
from ._compiled_element import CompiledElement
from ._compiled_interaction import CompiledInteraction
from ._form import Form
from ._layout import Layout
from ._network_snapshot import NetworkSnapshot
from ._renewal_record import RenewalRecord
from ._transfer_network import TransferNetwork
from ._variable import Variable

if TYPE_CHECKING:
    from ._partition import Partition

_ROW_TOLERANCE = 1e-9


type Getter = Callable[[float, Vector, Vector, Mapping[str, Any]], Any]


class _RenewalProcess(Protocol):
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


class _TopologyChanges(Protocol):
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

    def presence(self, name: str, p: Vector) -> npt.NDArray[np.bool_] | None:
        """Return whether every row of the group of a variable holds a member, or `None` for a fixed group."""


@dataclass(frozen=True)
class _MarkovMatrix:
    """Transition or emission matrix of a law, evaluated from the parameter vector to check its rows."""

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
            wrong = np.flatnonzero(np.abs(sums - 1) > _ROW_TOLERANCE)
            if wrong.size:
                return f'the row {wrong[0]}{where} of the {self.law} sums to {sums[wrong[0]]}, not 1'
        return None


@dataclass(frozen=True)
class BaseProblem:
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
        >>> from skcomplex.engine.problems import Form, symbol
        >>> from skcomplex.engine.problems import Layout
        >>> from skcomplex.engine.problems import BaseProblem
        >>> problem = BaseProblem(Form.ODE, Layout([]), Layout([]), np.zeros(0), np.zeros(0), symbol('t'))
        >>> problem.stochastic, problem.adaptive_parameters, problem.lattice_shape('a.n')
        (False, {}, None)
    """

    form: Form
    states: Layout
    parameters: Layout
    initial_state: Vector
    initial_parameters: Vector
    time: Expression
    _: KW_ONLY
    symbols: Mapping[str, Expression | FreeModuleElement] = field(default_factory=dict, repr=False)
    elements: tuple[CompiledElement, ...] = ()
    interactions: tuple[CompiledInteraction, ...] = ()
    system: System | None = field(default=None, repr=False)
    agents: tuple[CompiledAgents, ...] = ()
    agent_space: BaseSpace | None = None
    partitions: tuple['Partition', ...] = ()
    companions: Mapping[str, tuple[str, str, int]] = field(default_factory=dict)
    renewals: tuple[_RenewalProcess, ...] = field(default=(), repr=False)
    matrices: tuple[_MarkovMatrix, ...] = field(default=(), repr=False)
    transfer_network: TransferNetwork | None = field(default=None, repr=False)
    topology: _TopologyChanges | None = field(default=None, repr=False)

    @property
    def stochastic(self) -> bool:
        """Whether the problem draws random values, so its solver takes the random number generator."""
        return False

    @property
    def adaptive_parameters(self) -> dict[str, Variable]:
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
