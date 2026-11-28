"""Define the base problem spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from ...base import BaseSpec
from ...components import Observable
from ...components.elements import Element, VisualSpec
from ...components.interactions import BaseInteraction
from ...components.laws import BaseLaw
from ...components.spaces import BaseSpace
from ...enums import FormKind
from ...types import Array
from ._compiled_element_spec import CompiledElementSpec, _Variable
from ._composite_system import _CompositeSystem
from ._group_element import _GroupElement
from ._layout_spec import LayoutSpec
from ._renewal_kernel import RenewalKernel
from ._topology_kernel import TopologyKernel
from ._transfer_network_spec import TransferNetworkSpec


@dataclass(frozen=True)
class _CompiledAgents:
    """A record of an agent element or agent group of an agent-based problem.

    The states of its agents live in the Mesa model that runs them.

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

        objects:
            The names of the states that hold hashable objects.

        roles:
            A mapping from the name of a state with a role in a snapshot to its role.
    """

    path: str
    element: Element | _GroupElement
    size: int
    states: tuple[str, ...]
    values: Mapping[str, object]
    parameters: Mapping[str, str]
    rules: tuple[BaseLaw, ...]
    visual: VisualSpec
    objects: tuple[str, ...] = ()
    roles: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _CompiledInteraction:
    """A record of an interaction of a compiled problem, with the paths its roles bind.

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
    interaction: BaseInteraction
    roles: Mapping[str, str | tuple[str, ...]]


@dataclass(frozen=True)
class BaseProblemSpec(BaseSpec):
    """An abstract spec of a problem that a system compiles to.

    It holds what every solver, observer, and visual reads of a problem, such as the names of the states and their
    starting values, with empty defaults for what a form does not have, and its readers answer the questions of the run.
    A form that brings its own solver adds the fields that solver reads. It is the base of the problems of a growing
    population, a quantum circuit, and a market of agents.

    Read more in the user guide, under Extend the library.

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

        companions:
            A mapping from the name of every companion state to its kind, `'derivative'` or `'lag'`, the name of the
            state it belongs to, and the order of the derivative or the number of steps back.

        renewals:
            The renewal laws, whose members renew on their own clocks.


        transfer_network:
            The quantities, flows, objectives, and feedback terms of the system, or `None` without them.

        topology:
            The topology requests of the laws and the networks they change, or `None` for a problem without them.

        observables:
            The observables of the system tree, each with the path of the system that declares it, empty for the
            root, which the compiler collects.

    See Also:
        ProblemSpec: The problem of the general compiler.
        OperatorProblemSpec: The problem of operator systems.

    Examples:
        >>> from skcomplex.core.engine.problems import BaseProblemSpec
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> problem = LogisticGrowthSystem().prepare().simulate(until=0).problem
        >>> isinstance(problem, BaseProblemSpec), problem.states.names
        (True, ['population.x'])
    """

    form: FormKind | str
    states: LayoutSpec
    parameters: LayoutSpec
    initial_state: Array
    initial_parameters: Array
    time: Expression
    _: KW_ONLY
    symbols: Mapping[str, Expression | FreeModuleElement] = field(default_factory=dict, repr=False)
    elements: tuple[CompiledElementSpec, ...] = ()
    interactions: tuple[_CompiledInteraction, ...] = ()
    system: _CompositeSystem | None = field(default=None, repr=False)
    agents: tuple[_CompiledAgents, ...] = ()
    agent_space: BaseSpace | None = None
    companions: Mapping[str, tuple[str, str, int]] = field(default_factory=dict)
    renewals: tuple[RenewalKernel, ...] = field(default=(), repr=False)
    transfer_network: TransferNetworkSpec | None = field(default=None, repr=False)
    topology: TopologyKernel | None = field(default=None, repr=False)
    observables: tuple[tuple[str, Observable], ...] = field(default=(), repr=False)

    @property
    def partition_problems(self) -> tuple[BaseProblemSpec, ...]:
        """Problems of the partitions of a co-simulation problem.

        A problem of one form has none.
        """
        return ()

    @property
    def stochastic(self) -> bool:
        """Whether the problem draws random values.

        A problem that draws them has a solver that takes the random number generator.
        """
        return False

    @property
    def adaptive_parameters(self) -> dict[str, _Variable]:
        """Declarations of the adaptive parameters, by dotted state name.

        Adaptive parameters are states that a schedule may set.
        """
        return {}

    def get_lattice_shape(self, name: str) -> tuple[int, ...] | None:
        """Return the lattice shape of a state that lives on the sites of a lattice.

        Other states return `None`.

        Args:
            name:
                The dotted name of the state.

        Returns:
            shape:
                The number of sites along every dimension, or `None`. No state is on a lattice by default.
        """
        del name
        return None
