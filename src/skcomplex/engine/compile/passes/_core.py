"""List the passes, the checks, and the whole-system compilers that every compilation runs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.system import System
from .._pipeline import BasePass, Check
from .._system_compilers import BaseSystemCompiler
from ..checks import probability_issues, request_issues
from ._agents import AgentsPass
from ._agents_problem import AgentsProblemPass
from ._census import CensusPass
from ._check_fractional_noise import CheckFractionalNoisePass
from ._collect import CollectPass
from ._compile_encounters import CompileEncountersPass
from ._complex import LowerComplexPass
from ._conserved_noise import ConservedNoisePass, _conserved_issues
from ._declare_renewals import DeclareRenewalsPass
from ._draws import DrawsPass
from ._encounter_functions import EncounterFunctionsPass
from ._encounter_jacobian import EncounterJacobianPass
from ._encounter_problem import EncounterProblemPass
from ._events import DelaysPass
from ._expand_flows import ExpandFlowsPass
from ._field_functions import FieldFunctionsPass
from ._field_jacobian import FieldJacobianPass
from ._field_problem import FieldProblemPass
from ._flatten import FlattenPass
from ._flows_problem import FlowsProblemPass
from ._form import FormPass
from ._fractional_form import FractionalFormPass, _order_issues
from ._fractional_problem import FractionalProblemPass
from ._functions import FunctionsPass
from ._jacobian import JacobianPass
from ._kernels import KernelsPass
from ._lattices import LatticeProblemPass, _schedule_issues
from ._layout import LayoutPass
from ._lower_fields import LowerFieldsPass
from ._lower_flows import LowerFlowsPass
from ._markov import MarkovPass
from ._mask_requests import MaskRequestsPass
from ._members import MemberShiftPass
from ._noise import NoisePass
from ._orders import OrdersPass
from ._owners import OwnersPass
from ._particles import BoundaryLawsPass
from ._plan import PlanPass
from ._problem import ProblemPass
from ._quantum_compiler import QuantumCompiler
from ._reductions import ReductionsPass
from ._renewal_problem import RenewalProblemPass
from ._request_jumps import RequestJumpsPass
from ._rewriting_compiler import RewritingCompiler
from ._rows import RowsPass
from ._set_renewals_apart import SetRenewalsApartPass
from ._set_rules_apart import SetRulesApartPass
from ._specs import SpecsPass
from ._split_requests import SplitRequestsPass
from ._topology_functions import TopologyFunctionsPass
from ._topology_problem import TopologyProblemPass
from ._towards import TowardsPass
from ._unpack_matches import UnpackMatchesPass
from ._unpack_renewals import UnpackRenewalsPass

_PASSES: tuple[BasePass, ...] = (
    FlattenPass(),
    CensusPass(),
    PlanPass(),
    LayoutPass(),
    SpecsPass(),
    CollectPass(),
    LowerComplexPass(),
    MarkovPass(),
    RowsPass(),
    OrdersPass(),
    DelaysPass(),
    NoisePass(),
    MemberShiftPass(),
    OwnersPass(),
    DrawsPass(),
    FormPass(),
    ReductionsPass(),
    KernelsPass(),
    FunctionsPass(),
    JacobianPass(),
    ProblemPass(),
    UnpackRenewalsPass(),
    DeclareRenewalsPass(),
    SetRenewalsApartPass(),
    RenewalProblemPass(),
    ExpandFlowsPass(),
    LowerFlowsPass(),
    FlowsProblemPass(),
    BoundaryLawsPass(),
    AgentsPass(),
    UnpackMatchesPass(),
    SetRulesApartPass(),
    AgentsProblemPass(),
    LatticeProblemPass(),
    LowerFieldsPass(),
    ConservedNoisePass(),
    FieldFunctionsPass(),
    FieldJacobianPass(),
    FieldProblemPass(),
    CheckFractionalNoisePass(),
    FractionalFormPass(),
    FractionalProblemPass(),
    CompileEncountersPass(),
    EncounterFunctionsPass(),
    EncounterJacobianPass(),
    EncounterProblemPass(),
    SplitRequestsPass(),
    MaskRequestsPass(),
    TowardsPass(),
    RequestJumpsPass(),
    TopologyFunctionsPass(),
    TopologyProblemPass(),
)
"""Passes of a compilation, in the order that `order_passes` keeps where their fields leave them unordered."""

_CHECKS: tuple[Check, ...] = (
    _schedule_issues,
    _conserved_issues,
    _order_issues,
    probability_issues,
    request_issues,
)
"""Checks that validation runs on the compilation state of every system the general compiler compiles."""

_SYSTEM_COMPILERS: tuple[BaseSystemCompiler, ...] = (QuantumCompiler(), RewritingCompiler())
"""Compilers of the whole systems they claim, which compile them in place of the passes."""


def claiming_compiler(system: System) -> BaseSystemCompiler | None:
    """Return the first compiler that claims a system, or `None` when the general compiler compiles it.

    Args:
        system:
            The system.

    Returns:
        compiler:
            The compiler of the system, or `None`.

    Examples:
        >>> from skcomplex.engine.compile import claiming_compiler
        >>> from skcomplex.models import create
        >>> claiming_compiler(create('lotka_volterra')) is None
        True
    """
    for compiler in _SYSTEM_COMPILERS:
        if compiler.prepare(system).claims_:
            return compiler
    return None
