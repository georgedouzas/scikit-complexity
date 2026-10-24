"""List the passes of the shared core, and register its own passes, forms, owner rules, and bindings."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...base import Form
from .._census import register_owner
from .._forms import register_combined, register_form
from .._pipeline import register_pass
from .._specs import register_binding
from . import _flows, _particles, _waiting_jumps
from ._census import CensusPass
from ._collect import CollectPass
from ._complex import LowerComplexPass
from ._draws import DrawsPass
from ._events import DelaysPass
from ._flatten import FlattenPass
from ._form import FormPass
from ._functions import FunctionsPass
from ._jacobian import JacobianPass
from ._kernels import KernelsPass
from ._layout import LayoutPass
from ._markov import MarkovPass
from ._members import MemberShiftPass
from ._noise import NoisePass
from ._orders import OrdersPass
from ._owners import OwnersPass
from ._plan import PlanPass
from ._problem import ProblemPass
from ._reductions import ReductionsPass
from ._rows import RowsPass
from ._specs import SpecsPass

CORE_PASSES = (
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
    *_waiting_jumps.PASSES,
)
REGISTERED_PASSES = (
    *_flows.PASSES,
    *_particles.PASSES,
)
STEPWISE_FORMS: dict[frozenset[type], Form] = {
    **_waiting_jumps.STEPWISE_FORMS,
}
OWNER_RULES = (
    *_waiting_jumps.OWNER_RULES,
    *_flows.OWNER_RULES,
)


def register_shared_rules() -> None:
    """Register the passes, forms, owner rules, and bindings of stocks and flows, waiting-time jumps, and pairs."""
    for item in REGISTERED_PASSES:
        register_pass(item)
    for kinds, form in STEPWISE_FORMS.items():
        register_form(kinds, form, stepwise=True)
    for kind, companions in _waiting_jumps.COMBINED.items():
        register_combined(kind, companions)
    for rule in OWNER_RULES:
        register_owner(rule)
    for over, binding in _particles.BINDINGS.items():
        register_binding(over, binding)


register_shared_rules()
