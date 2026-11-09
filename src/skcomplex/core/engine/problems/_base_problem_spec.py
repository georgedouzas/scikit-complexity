"""Define the base problem spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from ...components.spaces import BaseSpace
from ...components.specs import CompiledAgentsSpec, CompiledElementSpec, CompiledInteractionSpec
from ...components.systems import CompositeSystem
from ...specs._base_spec import BaseSpec
from ...specs._layout_spec import LayoutSpec
from ...specs._observable_spec import ObservableSpec
from ...specs._variable_spec import VariableSpec
from ...types import Array, FormKind
from ._markov_kernel import MarkovKernel
from ._renewal_kernel import RenewalKernel
from ._topology_kernel import TopologyKernel
from ._transfer_network_spec import TransferNetworkSpec


@dataclass(frozen=True)
class BaseProblemSpec(BaseSpec):
    """Base of the problems a system compiles to.

    It holds what every solver, observer, and visual reads of a problem. A problem is a value. Its fields hold what the
    runtime reads, with trivial defaults for what a form does not have, and its readers answer the runtime's questions,
    trivially by default. A formalism's problem adds the fields its own solver reads.

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

        matrices:
            The transition and emission matrices of the Markov laws, checked when their parameters change.

        transfer_network:
            The quantities, flows, objectives, and feedback terms of the system, or `None` without them.

        topology:
            The topology requests of the laws and the networks they change, or `None` for a problem without them.

        observables:
            The observables of the system tree, each with the path of the system that declares it, empty for the
            root, which the compiler collects.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.types import FormKind
        >>> from skcomplex.core.engine._utils import symbol
        >>> from skcomplex.core.specs import LayoutSpec
        >>> from skcomplex.core.engine.problems import BaseProblemSpec
        >>> layout = LayoutSpec(())
        >>> problem = BaseProblemSpec(FormKind.ODE, layout, layout, np.zeros(0), np.zeros(0), symbol('t'))
        >>> problem.stochastic, problem.adaptive_parameters, problem.lattice_shape('a.n')
        (False, {}, None)
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
    interactions: tuple[CompiledInteractionSpec, ...] = ()
    system: CompositeSystem | None = field(default=None, repr=False)
    agents: tuple[CompiledAgentsSpec, ...] = ()
    agent_space: BaseSpace | None = None
    companions: Mapping[str, tuple[str, str, int]] = field(default_factory=dict)
    renewals: tuple[RenewalKernel, ...] = field(default=(), repr=False)
    matrices: tuple[MarkovKernel, ...] = field(default=(), repr=False)
    transfer_network: TransferNetworkSpec | None = field(default=None, repr=False)
    topology: TopologyKernel | None = field(default=None, repr=False)
    observables: tuple[tuple[str, ObservableSpec], ...] = field(default=(), repr=False)

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
    def adaptive_parameters(self) -> dict[str, VariableSpec]:
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
