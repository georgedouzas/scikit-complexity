"""List the passes, the checks, and the whole-system compilers that every compilation runs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.system import System
from .._pipeline import BasePass, Check
from .._system_compilers import BaseSystemCompiler
from ..checks import probability_issues, request_issues
from ._agents import AgentsPass, AgentsProblemPass, SetRulesApartPass, UnpackMatchesPass
from ._census import CensusPass
from ._collect import CollectPass
from ._complex import LowerComplexPass
from ._draws import DrawsPass
from ._events import DelaysPass
from ._fields import (
    ConservedNoisePass,
    FieldFunctionsPass,
    FieldJacobianPass,
    FieldProblemPass,
    LowerFieldsPass,
    conserved_issues,
)
from ._flatten import FlattenPass
from ._flows import ExpandFlowsPass, FlowsProblemPass, LowerFlowsPass
from ._form import FormPass
from ._fractional import CheckFractionalNoisePass, FractionalFormPass, FractionalProblemPass, order_issues
from ._functions import FunctionsPass
from ._jacobian import JacobianPass
from ._kernels import KernelsPass
from ._kinetics import CompileEncountersPass, EncounterFunctionsPass, EncounterJacobianPass, EncounterProblemPass
from ._lattices import LatticeProblemPass, schedule_issues
from ._layout import LayoutPass
from ._markov import MarkovPass
from ._members import MemberShiftPass
from ._noise import NoisePass
from ._orders import OrdersPass
from ._owners import OwnersPass
from ._particles import BoundaryLawsPass
from ._plan import PlanPass
from ._problem import ProblemPass
from ._quantum import QuantumCompiler
from ._reductions import ReductionsPass
from ._rewriting import RewritingCompiler
from ._rows import RowsPass
from ._specs import SpecsPass
from ._topology import (
    MaskRequestsPass,
    RequestJumpsPass,
    SplitRequestsPass,
    TopologyFunctionsPass,
    TopologyProblemPass,
    TowardsPass,
)
from ._waiting_jumps import DeclareRenewalsPass, RenewalProblemPass, SetRenewalsApartPass, UnpackRenewalsPass

PASSES: tuple[BasePass, ...] = (
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

CHECKS: tuple[Check, ...] = (
    schedule_issues,
    conserved_issues,
    order_issues,
    probability_issues,
    request_issues,
)
"""Checks that validation runs on the compilation state of every system the general compiler compiles."""

SYSTEM_COMPILERS: tuple[BaseSystemCompiler, ...] = (QuantumCompiler(), RewritingCompiler())
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
    for compiler in SYSTEM_COMPILERS:
        if compiler.prepare(system).claims_:
            return compiler
    return None
