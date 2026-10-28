"""Hold the state of one compilation, the fields its passes produce in order from the system to the problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ...core.laws import BaseLaw, CrossingLaw, RuleLaw
from ...core.specs import TopologyRequestSpec
from ...core.system import System
from ..backends import BaseBackend
from ..problems import Form, Lowered, Output, Problem, ReductionSchedule, TransferNetwork
from ..problems._base_problem import _MarkovMatrix
from ._block import Block
from ._bound_renewal import BoundRenewal
from ._census import Census
from ._compiled_encounter import CompiledEncounter
from ._declarations import Declarations
from ._diffusive_limit import DiffusiveLimit
from ._draw_lowering import DrawLowering
from ._field_splitting import LinearTerm
from ._field_volumes import FiniteVolume
from ._flows import TransferPass
from ._jacobians import ExactJacobian
from ._kernel_spec import KernelSpec
from ._matrix_law import MatrixLaw
from ._plan import _Plan
from ._topology import Topology
from ._tree import Node

type Laws = list[tuple[KernelSpec, list[BaseLaw]]]
type BlockLaws = list[tuple[KernelSpec, list[BaseLaw], Block]]

class Compilation:
    """State of one compilation of a system, whose fields its passes read and write in order.

    The constructor stores the inputs of the compilation, and the passes set every other field, from `nodes` to
    `problem` in the order they are listed. A field that no pass has set yet is `None`. A pass may also declare hidden
    states and parameters on the layout, which only grows.

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

        drawing:
            The draws of the laws.

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

        flows:
            The lowering of flows and delays, which one pass creates and a later pass finishes.

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

    nodes: list[Node] | None = None
    census: Census | None = None
    plan: _Plan | None = None
    layout: Declarations | None = None
    specs: list[tuple[KernelSpec, Block]] | None = None
    collected: BlockLaws | None = None
    apart: Laws | None = None
    lowered: BlockLaws | None = None
    markov: list[tuple[KernelSpec, MatrixLaw]] | None = None
    matrices: tuple[_MarkovMatrix, ...] | None = None
    ordered: Laws | None = None
    delayed: Laws | None = None
    corrected: Laws | None = None
    shifted: Laws | None = None
    drawn: Laws | None = None
    drawn_apart: Laws | None = None
    kinds: dict[str, set[type]] | None = None
    drawing: DrawLowering | None = None
    form: Form | None = None
    reduced: Laws | None = None
    reduced_apart: Laws | None = None
    reductions: ReductionSchedule | None = None
    kernels: Lowered | None = None
    functions: dict[str, Any] | None = None
    jacobian: ExactJacobian | None = None
    problem: Problem | None = None
    delays: list[dict[str, Any]] | None = None
    placeholders: dict[tuple[int, str, int, str], Expression] | None = None
    events: list[tuple[KernelSpec, CrossingLaw]] | None = None
    renewals: list[BoundRenewal] | None = None
    outputs: dict[int, list[Output]] | None = None
    written: dict[tuple[str, int], list[tuple[Expression, Expression]]] | None = None
    limits: list[float] | None = None
    volume_limits: list[DiffusiveLimit] | None = None
    linear: list[LinearTerm] | None = None
    volumes: tuple[FiniteVolume, ...] | None = None
    symmetries: tuple[tuple[npt.NDArray[np.intp], tuple[int, ...], list[tuple[int, int]]], ...] | None = None
    constrained: tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]] | None = None
    encounters: tuple[CompiledEncounter, ...] | None = None
    flows: TransferPass | None = None
    transfer_network: TransferNetwork | None = None
    rules: list[tuple[KernelSpec, tuple[RuleLaw, ...]]] | None = None
    request_jumps: dict[int, tuple[KernelSpec, TopologyRequestSpec]] | None = None
    topology: Topology | None = None
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
