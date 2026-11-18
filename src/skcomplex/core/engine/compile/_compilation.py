"""Hold the state of one compilation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression
from sage.symbolic.function import SymbolicFunction
from sage.symbolic.operators import FDerivativeOperator

from ...components.laws import BaseLaw, ConstraintLaw, EventLaw, ObjectiveLaw, RuleLaw, TopologyRequestSpec
from ...components.spaces import StaticTime
from ...components.systems import CompositeSystem
from ...enums import FormKind, VariableKind
from ...errors import PipelineError
from .._utils import declarations_of
from ..backends import BaseBackend
from ..enums import ReadKind
from ..problems import ProblemSpec
from ..problems._block_spec import BlockSpec
from ..problems._declarations_spec import DeclarationsSpec
from ..problems._delayed_state_spec import _DelayedStateSpec
from ..problems._diffusive_limit_spec import DiffusiveLimitSpec
from ..problems._draw_kernel import DrawKernel
from ..problems._encounter_kernel import EncounterKernel
from ..problems._finite_volume_kernel import FiniteVolumeKernel
from ..problems._jacobian_kernel import JacobianKernel
from ..problems._kernel_spec import KernelSpec
from ..problems._linear_term_spec import LinearTermSpec
from ..problems._lowered_kernels_spec import LoweredKernelsSpec
from ..problems._markov_kernel import MarkovKernel
from ..problems._node_spec import _NodeSpec
from ..problems._output_spec import OutputSpec
from ..problems._problem_functions_spec import _ProblemFunctionsSpec
from ..problems._reduction_schedule_spec import ReductionScheduleSpec
from ..problems._symmetry_spec import SymmetrySpec
from ..problems._topology_kernel import TopologyKernel
from ..problems._transfer_network_spec import TransferNetworkSpec
from ..problems._transfer_state_spec import TransferStateSpec
from ._bound_renewal_spec import BoundRenewalSpec
from ._census_spec import _CensusSpec
from ._matrix_law_spec import _MatrixLawSpec
from ._plan_spec import _PlanSpec

TIME_FUNCTIONS = ('derivative', 'lag', 'delayed')

type Laws = list[tuple[KernelSpec, list[BaseLaw]]]
type BlockLaws = list[tuple[KernelSpec, list[BaseLaw], BlockSpec]]


class Compilation:
    """A state of one compilation of a system, which passes read and write.

    Its passes read and write its fields in order. The constructor stores the inputs of the compilation, and the passes
    set every other field, from `nodes` to `problem` in the order they are listed. A field that no pass has set yet is
    `None`. A pass may also declare hidden states and parameters on the layout, which only grows. Kernel specs are
    frozen: a pass that adds references to a spec builds a copy, which `_replace_specs` puts in every field that holds
    the spec.

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

    Attributes:
        nodes:
            The elements, groups, and interactions of the system tree, flattened once.

        census:
            The laws every component declares, with their owners.

        plan:
            The partitions of the system.

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
            The laws of every kernel spec with complex states split and Markov laws taken out.

        markov:
            The Markov laws of every kernel spec.

        matrices:
            The Markov matrices, checked at the initial parameters.

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
    """

    nodes: list[_NodeSpec] | None = None
    census: _CensusSpec | None = None
    plan: _PlanSpec | None = None
    layout: DeclarationsSpec | None = None
    specs: list[tuple[KernelSpec, BlockSpec]] | None = None
    collected: BlockLaws | None = None
    apart: Laws | None = None
    programs: Laws | None = None
    lowered: BlockLaws | None = None
    markov: list[tuple[KernelSpec, _MatrixLawSpec]] | None = None
    matrices: tuple[MarkovKernel, ...] | None = None
    ordered: Laws | None = None
    delayed: Laws | None = None
    corrected: Laws | None = None
    shifted: Laws | None = None
    drawn: Laws | None = None
    drawn_apart: Laws | None = None
    kinds: dict[str, set[type]] | None = None
    draws: DrawKernel | None = None
    form: FormKind | None = None
    reduced: Laws | None = None
    reduced_apart: Laws | None = None
    reductions: ReductionScheduleSpec | None = None
    kernels: LoweredKernelsSpec | None = None
    functions: _ProblemFunctionsSpec | None = None
    jacobian: JacobianKernel | None = None
    problem: ProblemSpec | None = None
    delays: list[_DelayedStateSpec] | None = None
    placeholders: dict[tuple[str, int, str, int, str, ReadKind], Expression] | None = None
    events: list[tuple[KernelSpec, EventLaw]] | None = None
    renewals: list[BoundRenewalSpec] | None = None
    outputs: dict[int, list[OutputSpec]] | None = None
    written: dict[tuple[str, int], list[tuple[Expression, Expression]]] | None = None
    limits: list[float] | None = None
    volume_limits: list[DiffusiveLimitSpec] | None = None
    linear: list[LinearTermSpec] | None = None
    volumes: tuple[FiniteVolumeKernel, ...] | None = None
    symmetries: tuple[SymmetrySpec, ...] | None = None
    constrained: tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]] | None = None
    encounters: tuple[EncounterKernel, ...] | None = None
    transfer: TransferStateSpec | None = None
    transfer_network: TransferNetworkSpec | None = None
    rules: list[tuple[KernelSpec, tuple[RuleLaw, ...]]] | None = None
    request_jumps: dict[int, tuple[KernelSpec, TopologyRequestSpec]] | None = None
    topology: TopologyKernel | None = None
    done: set[object] | None = None

    def __init__(
        self,
        system: CompositeSystem,
        backend: BaseBackend | None = None,
        solver: object = None,
        owners: frozenset[str] | None = None,
        held: tuple[str, ...] = (),
    ) -> None:
        self.system = system
        self.backend = backend
        self.solver = solver
        self.owners = owners
        self.held = held

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
        self._replace_in(self.markov or [], replaced)
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
    def _has_motion(layout: DeclarationsSpec) -> bool:
        """Return whether a state of the layout has a derivative, as a point given a starting velocity has.

        Such a state moves at its derivative without any law, so its system is not static.
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

        It is static by its time, by an objective law, by constraint laws alone, or by having no laws at all.

        A system with no laws and no time of its own, such as `System(elements=[Element('ball', values={'x': 1.0})])`
        or disks placed in a square, changes nothing, so it is static and records one frame. A point given a starting
        velocity moves at it without a law, so its system is not static. A system whose laws
        are all constraints and that has no time of its own, such as the best responses of the firms of a Cournot game,
        is solved once, so it is static too.

        Returns:
            static:
                Whether the system has static time, an objective law, constraint laws alone, or no laws and no time.
        """
        if isinstance(self.system.time, StaticTime):
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
            and all(self._is_balance(record.law) for record in census.records)
        ):
            return True
        return any(isinstance(record.law, ObjectiveLaw) for record in census.records)
