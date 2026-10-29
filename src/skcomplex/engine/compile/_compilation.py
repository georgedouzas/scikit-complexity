"""Hold the state of one compilation, the fields its passes produce in order from the system to the problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ...core.laws import BaseLaw, CrossingLaw, RuleLaw
from ...core.specs import CensusSpec, NodeSpec
from ...core.system import System
from ...errors import PipelineError
from ...specs import MatrixLawSpec, TopologyRequestSpec, TransferNetworkSpec
from ...specs._plan_spec import _PlanSpec
from ...types import FormKind
from ..backends import BaseBackend
from ..problems import (
    BlockSpec,
    DeclarationsSpec,
    DiffusiveLimitSpec,
    KernelSpec,
    LinearTermSpec,
    LoweredKernelsSpec,
    OutputSpec,
    ProblemSpec,
    ReductionScheduleSpec,
    TransferStateSpec,
)
from ..problems._delayed_state_spec import _DelayedStateSpec
from ..problems._draw_kernel import DrawKernel
from ..problems._encounter_kernel import EncounterKernel
from ..problems._finite_volume_kernel import FiniteVolumeKernel
from ..problems._jacobian_kernel import JacobianKernel
from ..problems._markov_kernel import MarkovKernel
from ..problems._problem_functions_spec import _ProblemFunctionsSpec
from ..problems._topology_kernel import TopologyKernel
from ._bound_renewal_spec import BoundRenewalSpec

type Laws = list[tuple[KernelSpec, list[BaseLaw]]]
type BlockLaws = list[tuple[KernelSpec, list[BaseLaw], BlockSpec]]


class Compilation:
    """State of one compilation of a system, whose fields its passes read and write in order.

    The constructor stores the inputs of the compilation, and the passes set every other field, from `nodes` to
    `problem` in the order they are listed. A field that no pass has set yet is `None`. A pass may also declare hidden
    states and parameters on the layout, which only grows. Kernel specs are frozen: a pass that adds references to a
    spec builds a copy, which `_replace_specs` puts in every field that holds the spec.

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
            The state indices, the lattice shape, and the axes with their signs of every symmetric field state.

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

    nodes: list[NodeSpec] | None = None
    census: CensusSpec | None = None
    plan: _PlanSpec | None = None
    layout: DeclarationsSpec | None = None
    specs: list[tuple[KernelSpec, BlockSpec]] | None = None
    collected: BlockLaws | None = None
    apart: Laws | None = None
    lowered: BlockLaws | None = None
    markov: list[tuple[KernelSpec, MatrixLawSpec]] | None = None
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
    placeholders: dict[tuple[str, int, str, int, str], Expression] | None = None
    events: list[tuple[KernelSpec, CrossingLaw]] | None = None
    renewals: list[BoundRenewalSpec] | None = None
    outputs: dict[int, list[OutputSpec]] | None = None
    written: dict[tuple[str, int], list[tuple[Expression, Expression]]] | None = None
    limits: list[float] | None = None
    volume_limits: list[DiffusiveLimitSpec] | None = None
    linear: list[LinearTermSpec] | None = None
    volumes: tuple[FiniteVolumeKernel, ...] | None = None
    symmetries: tuple[tuple[npt.NDArray[np.intp], tuple[int, ...], list[tuple[int, int]]], ...] | None = None
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
        system: System,
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
            self.transfer = self.transfer.replaced(replaced)

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
    def _required[T](value: T | None, name: str) -> T:
        """Return the value of a field that a pass has set, raising an error for a field no pass has set yet.

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
