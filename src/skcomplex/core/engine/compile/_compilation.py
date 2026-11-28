"""Hold what one compilation of a system has found so far."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Self

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement
from sage.symbolic.function import SymbolicFunction
from sage.symbolic.operators import FDerivativeOperator

from ...components import EntrySpec
from ...components.base import BaseComponent
from ...components.elements import Element
from ...components.interactions import BaseInteraction, WienerProcessSpec
from ...components.laws import BaseLaw, ConstraintLaw, EventLaw, ObjectiveLaw, TopologyRequestSpec
from ...components.laws._linear_program import _LinearProgramLaw
from ...components.laws._rule import _RuleLaw
from ...components.spaces import BaseSpace
from ...components.spaces._static_time import _StaticTime
from ...enums import ElementKind, FormKind, VariableKind
from ...errors import PipelineError, SkcomplexError
from ...types import Array, ProcessKey, VectorFunction
from .._utils import declarations_of
from ..backends import BaseBackend
from ..enums import ReadKind
from ..problems import ProblemSpec
from ..problems._composite_system import _CompositeSystem
from ..problems._diffusive_limit_spec import DiffusiveLimitSpec
from ..problems._draw_kernel import DrawKernel
from ..problems._encounter_kernel import EncounterKernel
from ..problems._finite_volume_kernel import FiniteVolumeKernel
from ..problems._group_element import _GroupElement
from ..problems._jacobian_kernel import JacobianKernel, _Symmetry
from ..problems._kernel_spec import KernelSpec
from ..problems._law_kernel import _JumpChannels, _Output, _WienerBlock
from ..problems._linear_term_spec import LinearTermSpec
from ..problems._lowered_kernels_spec import LoweredKernelsSpec
from ..problems._neighborhood_kernel import _ReductionSchedule
from ..problems._problem_spec import _Accelerated
from ..problems._reference_spec import ReferenceSpec
from ..problems._topology_kernel import TopologyKernel
from ..problems._transfer_network_spec import TransferNetworkSpec, _CompiledDelay, _CompiledFlow
from ._issue_spec import IssueSpec
from ._renewal_law import RenewalLaw

TIME_FUNCTIONS = ('derivative', 'lag', 'delayed')
ACTIVE = '_active'
ENDS = ('_source', '_target')
MEMBERS = '_members'

type Laws = list[tuple[KernelSpec, list[BaseLaw]]]
type BlockLaws = list[tuple[KernelSpec, list[BaseLaw], _Block]]


@dataclass(frozen=True)
class _Node:
    """A record of an element or interaction with its place in the tree of a system.

    Args:
        path:
            The slash-separated path of the component.

        scope:
            The path prefix of the system that holds the component.

        component:
            The element, group, or interaction, or the Wiener processes that the noise role of an interaction binds.

        space:
            The nearest enclosing space.

        component_type:
            What declares the variables of the component, which `declarations_of` reads: the element type of a group,
            or the type of an element, an interaction, or a Wiener process, or the element or a member of the group for
            elements that declare their variables from data.

        members:
            The size of a group, or `None` for a single element, an interaction, or a Wiener process.

        interaction:
            Whether the component is not an element or a group: an interaction or a Wiener process.

        agent:
            Whether the component runs on the agent stepper, as a pass of a formalism decides.

        active:
            The initial number of members of a variable group on the sites or edges of a network, whose capacity is
            its size, or `None` for every other component.
    """

    path: str
    scope: str
    component: Element | _GroupElement | BaseInteraction | WienerProcessSpec
    space: BaseSpace | None
    component_type: type | BaseComponent
    members: int | None = None
    interaction: bool = False
    agent: bool = False
    active: int | None = None

    @property
    def as_element(self) -> Element | _GroupElement:
        """The component of a node of an element or a group.

        Raises:
            TypeError:
                The component is an interaction or a Wiener process.
        """
        if not isinstance(self.component, Element | _GroupElement):
            error_msg = f'The node {self.path} holds a {type(self.component).__name__}, not an element or a group.'
            raise TypeError(error_msg)
        return self.component

    @property
    def as_group(self) -> _GroupElement:
        """The component of a node of a group.

        Raises:
            TypeError:
                The component is not a group.
        """
        if not isinstance(self.component, _GroupElement):
            error_msg = f'The node {self.path} holds a {type(self.component).__name__}, not a group.'
            raise TypeError(error_msg)
        return self.component

    @property
    def as_interaction(self) -> BaseInteraction:
        """The component of a node of an interaction.

        Raises:
            TypeError:
                The component is an element, a group, or a Wiener process.
        """
        if not isinstance(self.component, BaseInteraction):
            error_msg = f'The node {self.path} holds a {type(self.component).__name__}, not an interaction.'
            raise TypeError(error_msg)
        return self.component

    @property
    def is_agent(self) -> bool:
        """Whether the node is an agent element or an agent group.

        Such a node runs on the agent stepper.
        """
        return self.agent


@dataclass(frozen=True)
class _Block:
    """A record of the values of one component and the symbols that stand for them.

    While it compiles, the engine gives every value of an element, group, or interaction a symbol for one member, such
    as `x_population` for the size of a population, and the entry where the value sits in the vectors of the run. Laws
    are written in these symbols and read back through this spec. It holds the symbols of a population, a reactor, and a
    market.

    Read more in the user guide, under Extend the library.

    Args:
        node:
            The component with its place in the system tree.

        symbols:
            The symbol, or the vector of symbols, of every variable of one member by name.

        entries:
            The layout entry of every variable by name.

        references:
            The component of a state or parameter every symbol stands for.

        per:
            The names of the parameters with one value per entry of a variadic role.

    See Also:
        _Declarations: Holds the block of every component.
        ReferenceSpec: Says which value a symbol stands for.

    Examples:
        >>> from dataclasses import fields
        >>> from skcomplex.core.engine.compile._compilation import _Block
        >>> [field.name for field in fields(_Block)]
        ['node', 'symbols', 'entries', 'references', 'per']
    """

    node: _Node
    symbols: Mapping[str, Expression | FreeModuleElement] = field(default_factory=dict)
    entries: Mapping[str, EntrySpec] = field(default_factory=dict)
    references: Mapping[Expression, ReferenceSpec] = field(default_factory=dict)
    per: frozenset[str] = frozenset()

    @property
    def rows(self) -> int:
        """Number of members, 1 for a single element or an interaction."""
        return self.node.members or 1


@dataclass(frozen=True)
class _Declarations:
    """A record of every value the components of a system declare, laid out for a run.

    While it compiles, the engine lists the states and parameters of every component in the order of the vectors of the
    run, with their initial values and the symbols that stand for them, such as the size of a population at index 0 of
    the states. Later passes of the compiler read the values from it. It lays out the values of an epidemic, a reactor,
    and a market.

    Read more in the user guide, under Extend the library.

    Args:
        time:
            The time symbol.

        states:
            The entries of the states, in the order of the state vector.

        parameters:
            The entries of the parameters, in the order of the parameter vector.

        initial_states:
            The flat initial value of every state entry.

        initial_parameters:
            The flat initial value of every parameter entry.

        blocks:
            The block of every element, group, and interaction by path.

        processes:
            The block of every Wiener process that the noise roles of interactions bind, by path.

        handles:
            The path of the Wiener process and the component of every handle symbol.

        agents:
            The path of the agent group and the variable of every symbol of agents.

    See Also:
        _Block: The values and symbols of one component.
        EntrySpec: Where one value sits in a vector.

    Examples:
        >>> from skcomplex.core.engine._utils import symbol
        >>> from skcomplex.core.engine.compile._compilation import _Declarations
        >>> _Declarations(symbol('t')).states
        ()
    """

    time: Expression
    states: tuple[EntrySpec, ...] = ()
    parameters: tuple[EntrySpec, ...] = ()
    initial_states: tuple[Array, ...] = ()
    initial_parameters: tuple[Array, ...] = ()
    blocks: Mapping[str, _Block] = field(default_factory=dict)
    processes: Mapping[str, _Block] = field(default_factory=dict)
    handles: Mapping[Expression, tuple[str, int]] = field(default_factory=dict)
    agents: Mapping[Expression, tuple[str, str]] = field(default_factory=dict)

    @property
    def ends(self) -> dict[bool, int]:
        """Sizes of the state vector and of the parameter vector.

        The size of the state vector is under `True`, and that of the parameter vector under `False`.
        """
        return {True: sum(entry.size for entry in self.states), False: sum(entry.size for entry in self.parameters)}

    @property
    def wieners(self) -> dict[str, WienerProcessSpec]:
        """Wiener processes that the noise roles of interactions bind, by path."""
        return {
            path: block.node.component
            for path, block in self.processes.items()
            if isinstance(block.node.component, WienerProcessSpec)
        }

    @property
    def members(self) -> int:
        """Number of members of the groups, leaving out agents.

        The number chooses a default backend.
        """
        return sum(block.node.members or 0 for block in self.blocks.values() if not block.node.is_agent)

    @cached_property
    def capacity(self) -> dict[str, _Block]:
        """Blocks of the variable groups on the sites or edges of networks, by path."""
        return {path: block for path, block in self.blocks.items() if block.node.active is not None}

    @cached_property
    def groups(self) -> frozenset[str]:
        """Paths of the declared groups."""
        return frozenset(path for path, block in self.blocks.items() if block.node.members is not None)

    def get_changing_edges(self, space: object) -> dict[str | None, _Block]:
        """Return the variable edge group of every layer of a network whose edges change.

        Args:
            space:
                The network space.

        Returns:
            groups:
                The block of the variable edge group by layer.
        """
        return {
            getattr(block.node.component, 'layer', None): block
            for block in self.capacity.values()
            if getattr(block.node.component, 'on', None) == 'edges' and block.node.space is space
        }

    def get_changing_entries(self, space: object) -> dict[str | None, tuple[EntrySpec, EntrySpec, EntrySpec]]:
        """Return the source, target, and activity entries of every changing layer.

        The entries are those of the variable edge group of the layer.

        Args:
            space:
                The network space.

        Returns:
            entries:
                The entries by layer.
        """
        return {
            layer: (block.entries[ENDS[0]], block.entries[ENDS[1]], block.entries[ACTIVE])
            for layer, block in self.get_changing_edges(space).items()
        }


@dataclass(frozen=True)
class _DelayedState:
    """State a law reads with a delay.

    The law reads it through the hidden parameter that solvers fill from its history.

    Args:
        entry:
            The entry of the hidden parameter.

        states:
            The indices of the delayed state in the state vector.

        tau:
            The delay.

        minimum:
            The smallest value of the delay.

        spec:
            The kernel spec of the law that reads the delayed state.

        symbol:
            The symbol of the hidden parameter in the law.

        state:
            The symbol of the delayed state.
    """

    entry: EntrySpec
    states: npt.NDArray[np.intp]
    tau: Expression
    minimum: float
    spec: KernelSpec
    symbol: Expression
    state: Expression


@dataclass(frozen=True)
class _ProblemFunctions:
    """Numeric functions, equations, and backends of a problem.

    The passes lower them and the problem takes them.

    Args:
        equations:
            The equation of every state, by state name.

        equation_symbols:
            The symbol of every state the equations read, by state name.

        has_callables:
            Whether a law of the form holds a Python callable.

        has_group_sums:
            Whether a law of the form sums over the members of a group.

        drift:
            The drift function, or `None`.

        sources:
            The part of the drift without the flux differences of finite-volume fields, or `None` without fluxes.

        diffusion:
            The function that returns every Wiener process amplitude, or `None`.

        noise_rows:
            The state index each noise amplitude drives.

        noise_columns:
            The Wiener process each noise amplitude multiplies.

        n_processes:
            The number of Wiener processes.

        noise_blocks:
            The blocks of correlated Wiener processes.

        propensities:
            The propensity function of the jumps, or `None`.

        jumps:
            The channels of the jumps.

        update:
            The update of a map, which takes a generator when the map draws random numbers, or `None`.

        boundary:
            The boundary laws, or `None`.

        algebraic:
            The function that sets the algebraic states of a state vector, or `None`.

        draws:
            The draws of the update laws, or `None`.

        backend:
            The name of the backend that lowered the functions.

        backends:
            The backend that runs the form and every law kind.

        accelerated:
            The compiled loops of the form, or `None`.
    """

    equations: Mapping[str, Expression]
    equation_symbols: Mapping[str, Expression]
    has_callables: bool
    has_group_sums: bool
    drift: VectorFunction | None = None
    sources: VectorFunction | None = None
    diffusion: VectorFunction | None = None
    noise_rows: npt.NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    noise_columns: npt.NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    n_processes: int = 0
    noise_blocks: tuple[_WienerBlock, ...] = ()
    propensities: VectorFunction | None = None
    jumps: tuple[_JumpChannels, ...] = ()
    update: Callable[..., Array] | None = None
    boundary: VectorFunction | None = None
    algebraic: VectorFunction | None = None
    draws: DrawKernel | None = None
    backend: str = 'numpy'
    backends: Mapping[str, str] = field(default_factory=dict)
    accelerated: _Accelerated | None = None


@dataclass(frozen=True)
class _PlannedPartition:
    """Subsystem that runs as one partition, by the scope of the laws it holds."""

    name: str
    scope: str
    nested: bool

    def holds(self, owner: str) -> bool:
        """Check whether the partition holds the laws of an element or group.

        Args:
            owner:
                The path of the element or group.

        Returns:
            holds:
                Whether the owner lies in the partition's subsystem, or in a system nested in it for a nested partition.
        """
        return owner.startswith(self.scope) and (self.nested or '/' not in owner[len(self.scope) :])


@dataclass(frozen=True)
class _CensusRecord:
    """One law of a component, with the element and the states it changes.

    Args:
        path:
            The path of the component that returns the law.

        owner:
            The path of the element or group whose state the law changes, or that a rule acts for.

        law:
            The law.

        targets:
            The layout entries of the states the law targets.

        vectors:
            The names of the targets declared as vector states.

        component:
            The kind of the owner: an element, a fixed or variable group, or a group on the sites of a space.

        space:
            The nearest space enclosing the component.

        processes:
            For a noise law, the Wiener process of every target component: `None` for an independent process,
            `('local', path, label)` for a local label, or `('wiener', path, index)` for a process of a
            `WienerProcessSpec`.

        driven:
            For a noise law, the state entry name and column of every target component.

        reads:
            For a noise law, the state components that the amplitude of every target component depends on.

        members:
            The number of members the law is evaluated for.

        handles:
            The handles of Wiener processes that the law uses outside the process of a noise law.

        degrees:
            The degree of the cells of every symbol the law reads of a group on a cell complex, by the name of the
            symbol.

        differentiated:
            For an equation, the paths of the elements and groups whose states it reads a derivative of, such as the
            cart of `(M + m) x'' + m l theta'' cos(theta) == F`.
    """

    path: str
    owner: str
    law: BaseLaw
    targets: tuple[EntrySpec, ...]
    vectors: frozenset[str]
    component: ElementKind
    space: BaseSpace | None
    processes: tuple[ProcessKey | None, ...] = ()
    driven: tuple[tuple[str, int], ...] = ()
    reads: tuple[frozenset[tuple[str, int]], ...] = ()
    members: int = 1
    handles: tuple[str, ...] = ()
    degrees: Mapping[str, int] = field(default_factory=dict, compare=False)
    differentiated: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _Census:
    """Laws and errors of every component of a system.

    A component fails when it cannot declare or collect its laws, and the census keeps its error instead of its laws.

    Args:
        system:
            The system.

        records:
            The laws of the components that declare and collect without error.

        errors:
            The path of every failing component with its error.

        agents:
            The path of the agent group and the variable of every symbol that reduces agents, an empty variable for
            the symbol of the members.
    """

    system: _CompositeSystem
    records: list[_CensusRecord]
    errors: list[tuple[str, SkcomplexError]]
    agents: Mapping[Expression, tuple[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class _BoundRenewal:
    """Renewal law with the group it renews and the hidden state that counts it.

    Args:
        spec:
            The kernel spec of the component.

        law:
            The renewal law.

        label:
            The label of the renewal.

        group:
            The path of the element or group the renewal resets.

        entry:
            The hidden state that counts the renewals of every member.

        symbol:
            The symbol of the count.
    """

    spec: KernelSpec
    law: RenewalLaw
    label: str
    group: str
    entry: EntrySpec
    symbol: Expression


@dataclass(frozen=True)
class _Born:
    """Transfer of a spec with the amount it moves and its compiled holders."""

    spec: KernelSpec
    flow: _CompiledFlow
    amount: Expression
    sign: int


@dataclass(frozen=True)
class _FlowTerm:
    """Flow or rate of a spec, with its source and sink.

    It becomes a node of the feedback graph.

    Args:
        spec:
            The spec whose references the expression reads.

        kind:
            `'flow'`, or `'rate'` for a rate law outside flows.

        path:
            The node of a flow, or the state a rate law targets.

        expression:
            The expression.

        source:
            The state a flow drains, or `None`.

        sink:
            The state a flow fills, or `None`.
    """

    spec: KernelSpec
    kind: str
    path: str
    expression: Expression
    source: str | None = None
    sink: str | None = None


@dataclass(frozen=True)
class _Record:
    """Recorded expression of a spec.

    It is rewritten as later passes substitute hidden states.
    """

    path: str
    expression: Expression
    spec: KernelSpec


@dataclass(frozen=True)
class _TransferState:
    """A record of the flows of a system halfway through compiling them.

    The compiler first expands every flow of a system into the laws it becomes, such as a flow from one tank to another
    that drains one and fills the other, then lowers them. It carries what lies between the two passes: the
    quantity of every state, the expanded flows, their records, the delays, and the terms of the feedback graph. It
    compiles the flows of water between tanks, money between accounts, and people between the compartments of an
    epidemic.

    Read more in the user guide, under Extend the library.

    Args:
        quantities:
            The quantity of every tagged state by entry name.

        born:
            The expanded flow of every law a flow became, by the identity of the law.

        flows:
            The expanded flows, in the order they were expanded.

        records:
            The recorded expressions of the flows.

        delays:
            The delays and smoothings lowered to hidden states.

        terms:
            The flows and rates of every spec, with their sources and sinks, for the loop terms.

    See Also:
        TransferNetworkSpec: What the lowering of flows gives.

    Examples:
        >>> from skcomplex.core.engine.compile._compilation import _TransferState
        >>> _TransferState({}, {}, (), (), (), ()).flows
        ()
    """

    quantities: Mapping[str, str]
    born: Mapping[int, _Born]
    flows: tuple[_Born, ...]
    records: tuple[_Record, ...]
    delays: tuple[_CompiledDelay, ...]
    terms: tuple[_FlowTerm, ...]

    def get_replaced(self, specs: Mapping[int, KernelSpec]) -> Self:
        """Return the state with replaced kernel specs.

        An example is specs whose rows are settled.

        Args:
            specs:
                The new spec of every replaced spec, by the identity of the replaced spec.

        Returns:
            state:
                The state that holds the new specs.
        """
        born = {id(item): replace(item, spec=specs.get(id(item.spec), item.spec)) for item in self.flows}
        return replace(
            self,
            born={key: born[id(item)] for key, item in self.born.items()},
            flows=tuple(born[id(item)] for item in self.flows),
            records=tuple(replace(record, spec=specs.get(id(record.spec), record.spec)) for record in self.records),
            terms=tuple(replace(term, spec=specs.get(id(term.spec), term.spec)) for term in self.terms),
        )


class Compilation:
    """A record of one compilation of a system, which passes read and write.

    The constructor stores the system and its settings, and the passes fill in every other field in turn, from `nodes`,
    the components of the system in a flat list, to `problem`, the finished problem. A field that no pass has set yet is
    `None`. For a population model the compilation holds the states of its elements in `layout`, their laws in
    `collected`, and the equations in `functions`. A pass may also add hidden states and parameters to the layout, which
    only grows.

    Read more in the user guide, under Extend the library.

    Args:
        system:
            The system, with the observables of the run.

        backend:
            The backend instance, or `None` for the backend that suits the size of the system.

        solver:
            The solver instance, or `None` for the default solver of the system's form.

        owners:
            The paths of the elements and groups whose laws a partition keeps, or `None` to keep every law.

        held:
            The paths of the observables that roles bind to and laws read only at synchronisation points.

        trained:
            The dotted paths of the parameters a training trains, such as `recurrence.A`, which the compilation makes
            adaptive states.

    Attributes:
        nodes:
            The elements, groups, and interactions of the system tree, flattened once.

        census:
            The laws every component declares, with their owners.

        plan:
            The partitions of the system and the issues of the law mixes that no form or solver takes.

        layout:
            The states and parameters of every component, with the blocks of their symbols.

        specs:
            The kernel spec of every component with its block.

        collected:
            The laws of every kernel spec as declared, with its block.

        apart:
            The laws set apart from the kernels of their spec, which a pass of their own compiles.

        programs:
            The linear program laws of every kernel spec, which never reach a kernel of their spec.

        lowered:
            The laws of every kernel spec with complex states split.

        ordered:
            The laws with derivatives and lags read through companion states and equations solved.

        delayed:
            The laws with delayed states read through hidden parameters and events taken out.

        corrected:
            The laws with the drift correction of every noise law of a convention.

        shifted:
            The laws with member offset reads lowered to neighbour reads.

        drawn:
            The laws of the owners of the compilation, masked to the rows present, with draws read from parameters.

        drawn_apart:
            The laws set apart of the owners of the compilation.

        kinds:
            The law classes of every owner.

        draws:
            The kernel of the draws of the laws.

        form:
            The form of the problem.

        reduced:
            The laws with every reduction read through its placeholder.

        reduced_apart:
            The laws set apart, with every reduction read through its placeholder.

        reductions:
            The reductions the laws read.

        kernels:
            The kernels of every law kind.

        functions:
            The numeric functions, the equations, and the backends of the problem.

        jacobian:
            The exact Jacobian of the drift, or `None`.

        problem:
            The compiled problem.

        delays:
            The delayed states with their hidden parameters.

        placeholders:
            The placeholder of every delayed state read.

        events:
            The events of every kernel spec.

        renewals:
            The renewal laws with the hidden states that count them.

        outputs:
            The outputs of the laws that a pass of a formalism lowers itself, by the identity of the law, which the
            kernels take in place of the outputs of the law's value.

        written:
            The operators of fields as written, by state and column, with their stencils.

        limits:
            The stable steps of explicit field operators.

        volume_limits:
            The stable steps of finite volumes, as functions of the parameters.

        linear:
            The linear terms with constant coefficients of the rates of field states.

        volumes:
            The conservation laws of fields.

        symmetries:
            The mirror symmetries of the field states.

        constrained:
            The indices and values of the fixed nodes of meshes.

        encounters:
            The encounter laws, lowered for their members.

        transfer:
            The state of the lowering of flows and delays, which one pass creates and a later pass finishes.

        transfer_network:
            The flows and stocks of the system.

        rules:
            The rules of every kernel spec, taken out of the reduced laws.

        request_jumps:
            The jumps that request topology changes, by the identity of the law, with their spec and request.

        topology:
            The topology changes of the networks, or `None`.

        done:
            The passes that ran.

    See Also:
        BasePass: The base of the steps that read and write a compilation.
        PassCompiler: Runs the passes over a compilation.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.biology.ecology import LotkaVolterraSystem
        >>> compilation = PassCompiler().prepare(LotkaVolterraSystem()).compilation_
        >>> [entry.name for entry in compilation.layout.states]
        ['prey.x', 'predators.x']
    """

    nodes: list[_Node] | None = None
    census: _Census | None = None
    plan: tuple[tuple[_PlannedPartition, ...], tuple[IssueSpec, ...]] | None = None
    layout: _Declarations | None = None
    specs: list[tuple[KernelSpec, _Block]] | None = None
    collected: BlockLaws | None = None
    apart: Laws | None = None
    programs: Laws | None = None
    lowered: BlockLaws | None = None
    ordered: Laws | None = None
    delayed: Laws | None = None
    corrected: Laws | None = None
    shifted: Laws | None = None
    drawn: Laws | None = None
    drawn_apart: Laws | None = None
    kinds: dict[str, set[type[BaseLaw]]] | None = None
    draws: DrawKernel | None = None
    form: FormKind | None = None
    reduced: Laws | None = None
    reduced_apart: Laws | None = None
    reductions: _ReductionSchedule | None = None
    kernels: LoweredKernelsSpec | None = None
    functions: _ProblemFunctions | None = None
    jacobian: JacobianKernel | None = None
    problem: ProblemSpec | None = None
    delays: list[_DelayedState] | None = None
    placeholders: dict[tuple[str, int, str, int, str, ReadKind], Expression] | None = None
    events: list[tuple[KernelSpec, EventLaw]] | None = None
    renewals: list[_BoundRenewal] | None = None
    outputs: dict[int, list[_Output]] | None = None
    written: dict[tuple[str, int], list[tuple[Expression, Expression]]] | None = None
    limits: list[float] | None = None
    volume_limits: list[DiffusiveLimitSpec] | None = None
    linear: list[LinearTermSpec] | None = None
    volumes: tuple[FiniteVolumeKernel, ...] | None = None
    symmetries: tuple[_Symmetry, ...] | None = None
    constrained: tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]] | None = None
    encounters: tuple[EncounterKernel, ...] | None = None
    transfer: _TransferState | None = None
    transfer_network: TransferNetworkSpec | None = None
    rules: list[tuple[KernelSpec, tuple[_RuleLaw, ...]]] | None = None
    request_jumps: dict[int, tuple[KernelSpec, TopologyRequestSpec]] | None = None
    topology: TopologyKernel | None = None
    done: set[object] | None = None

    def __init__(
        self,
        system: _CompositeSystem,
        backend: BaseBackend | None = None,
        solver: object = None,
        owners: frozenset[str] | None = None,
        held: tuple[str, ...] = (),
        *,
        trained: tuple[str, ...] = (),
    ) -> None:
        self.system = system
        self.backend = backend
        self.solver = solver
        self.owners = owners
        self.held = held
        self.trained = trained

    def _replace_specs(self, replaced: Mapping[int, KernelSpec]) -> None:
        """Replace kernel specs by their copies in every field that holds them.

        The fields keep their lists, so a pass that loops over a field reads the copies.

        Args:
            replaced:
                The copy of every replaced spec, by the identity of the replaced spec.
        """
        if not replaced:
            return
        self._replace_in(self.specs or [], replaced)
        self._replace_in(self.events or [], replaced)
        self._replace_in(self.rules or [], replaced)
        for block_laws in (self.collected, self.lowered):
            self._replace_in(block_laws or [], replaced)
        laws = (
            self.apart,
            self.programs,
            self.ordered,
            self.delayed,
            self.corrected,
            self.shifted,
            self.drawn,
            self.drawn_apart,
            self.reduced,
            self.reduced_apart,
        )
        for items in laws:
            self._replace_in(items or [], replaced)
        if self.delays is not None:
            self.delays[:] = [replace(delay, spec=replaced.get(id(delay.spec), delay.spec)) for delay in self.delays]
        if self.renewals is not None:
            self.renewals[:] = [
                replace(renewal, spec=replaced.get(id(renewal.spec), renewal.spec)) for renewal in self.renewals
            ]
        if self.request_jumps is not None:
            for key, (spec, request) in self.request_jumps.items():
                self.request_jumps[key] = (replaced.get(id(spec), spec), request)
        if self.transfer is not None:
            self.transfer = self.transfer.get_replaced(replaced)

    @staticmethod
    def _replace_in[*Rest](items: list[tuple[KernelSpec, *Rest]], replaced: Mapping[int, KernelSpec]) -> None:
        """Replace the kernel spec that leads every item of a field by its copy, in place.

        Args:
            items:
                The items of the field, each led by a kernel spec.

            replaced:
                The copy of every replaced spec, by the identity of the replaced spec.
        """
        for index, item in enumerate(items):
            spec = item[0]
            items[index] = (replaced.get(id(spec), spec), *item[1:])

    @staticmethod
    def _get_required[T](value: T | None, name: str) -> T:
        """Return the value of a field that a pass has set.

        A field that no pass has set yet raises an error.

        Args:
            value:
                The value of the field.

            name:
                The name of the field.

        Returns:
            value:
                The value, which is not `None`.

        Raises:
            PipelineError:
                No pass has set the field yet.
        """
        if value is None:
            error_msg = f'No pass has set the field {name!r} of the compilation yet. Run the pass that writes it first.'
            raise PipelineError(error_msg)
        return value

    @staticmethod
    def _has_motion(layout: _Declarations) -> bool:
        """Return whether a state of the layout has a derivative, as a moving point does.

        A point given a starting velocity has one. Such a state moves at its derivative without any law, so its system
        is not static.
        """
        return any(
            declaration.derivative_of is not None
            for block in layout.blocks.values()
            if not block.node.interaction
            for declaration in declarations_of(block.node.component_type, VariableKind.STATE).values()
        )

    @staticmethod
    def _is_balance(law: BaseLaw) -> bool:
        """Return whether a law is a constraint free of the time and of derivatives.

        Such a constraint, such as the best response `q1 == (a - q2) / 2` of a firm, holds at one moment only, while
        a source `v == E * sin(t)` or a circuit `L * q.diff(t, 2) + q / C == V` changes with time.
        """
        if not isinstance(law, ConstraintLaw):
            return False
        relation = SR(law.lhs) - SR(0 if law.rhs is None else law.rhs)
        return not Compilation._reads_time(relation)

    @staticmethod
    def _reads_time(expression: Expression) -> bool:
        """Return whether an expression reads a derivative or the time itself."""
        operator = expression.operator()
        if isinstance(operator, FDerivativeOperator):
            return True
        name = getattr(operator, 'name', None)
        if callable(name) and name() in TIME_FUNCTIONS:
            return True
        if isinstance(operator, SymbolicFunction):
            return False
        if operator is None:
            return expression.is_symbol() and str(expression) == 't'
        return any(Compilation._reads_time(operand) for operand in expression.operands())

    def is_static(self) -> bool:
        """Return whether the system is static.

        It is static by its time, by an objective law without controls, by constraint laws and linear programs alone,
        or by having no laws at all. An objective law with controls steers a run in time, and a linear program beside
        rate or update laws is solved at every time of the run.

        A system with no laws and no time of its own, such as `System(elements=[Element('ball', values={'x': 1.0})])`
        or disks placed in a square, changes nothing, so it is static and records one frame. A point given a starting
        velocity moves at it without a law, so its system is not static. A system whose laws
        are all constraints and that has no time of its own, such as the best responses of the firms of a Cournot game,
        is solved once, so it is static too.

        Returns:
            static:
                Whether the system has static time, an objective law without controls, constraint laws alone, or no
                laws and no time.
        """
        if isinstance(self.system.time, _StaticTime):
            return True
        census = self.census
        if census is None:
            return False
        unchanged = self.layout is not None and not self.layout.agents and not self._has_motion(self.layout)
        if not census.records and unchanged and self.system.time is None:
            return True
        if (
            census.records
            and self.system.time is None
            and all(
                isinstance(record.law, _LinearProgramLaw) or self._is_balance(record.law) for record in census.records
            )
        ):
            return True
        return any(isinstance(record.law, ObjectiveLaw) and record.law.controls is None for record in census.records)
