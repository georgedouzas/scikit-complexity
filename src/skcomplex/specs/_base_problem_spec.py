"""Define the base problem spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING, Self

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from ..errors import ValidationError
from ..types import Array, Getter
from ._base_spec import BaseSpec
from ._observable_spec import ObservableSpec

if TYPE_CHECKING:
    from ..core.spaces import BaseSpace
    from ..core.system import System
    from ..engine.problems._markov_kernel import MarkovKernel
    from ..engine.problems._renewal_kernel import RenewalKernel
    from ..engine.problems._topology_kernel import TopologyKernel
    from ..types import FormKind
    from ._compiled_agents_spec import CompiledAgentsSpec
    from ._compiled_element_spec import CompiledElementSpec
    from ._compiled_interaction_spec import CompiledInteractionSpec
    from ._layout_spec import LayoutSpec
    from ._partition_spec import PartitionSpec
    from ._transfer_network_spec import TransferNetworkSpec
    from ._variable_spec import VariableSpec


@dataclass(frozen=True)
class BaseProblemSpec(BaseSpec):
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
        >>> from skcomplex.types import FormKind
        >>> from skcomplex.engine._utils import symbol
        >>> from skcomplex.specs import LayoutSpec
        >>> from skcomplex.specs import BaseProblemSpec
        >>> problem = BaseProblemSpec(FormKind.ODE, LayoutSpec(()), LayoutSpec(()), np.zeros(0), np.zeros(0), symbol('t'))
        >>> problem.stochastic, problem.adaptive_parameters, problem.lattice_shape('a.n')
        (False, {}, None)
    """

    form: FormKind
    states: LayoutSpec
    parameters: LayoutSpec
    initial_state: Array
    initial_parameters: Array
    time: Expression
    _: KW_ONLY
    symbols: Mapping[str, Expression | FreeModuleElement] = field(default_factory=dict, repr=False)
    elements: tuple[CompiledElementSpec, ...] = ()
    interactions: tuple[CompiledInteractionSpec, ...] = ()
    system: System | None = field(default=None, repr=False)
    agents: tuple[CompiledAgentsSpec, ...] = ()
    agent_space: BaseSpace | None = None
    partitions: tuple[PartitionSpec, ...] = ()
    companions: Mapping[str, tuple[str, str, int]] = field(default_factory=dict)
    renewals: tuple[RenewalKernel, ...] = field(default=(), repr=False)
    matrices: tuple[MarkovKernel, ...] = field(default=(), repr=False)
    transfer_network: TransferNetworkSpec | None = field(default=None, repr=False)
    topology: TopologyKernel | None = field(default=None, repr=False)

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
    def adaptive_parameters(self) -> dict[str, VariableSpec]:
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
