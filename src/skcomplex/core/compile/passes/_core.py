"""List the passes of the shared core, and register the passes of the formalisms that still compile inside it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...base import Form
from .._census import register_owner
from .._forms import register_combined, register_form
from .._pipeline import register_pass
from .._specs import register_binding
from . import _agents, _control, _fields, _flows, _kinetics, _lattices, _particles, _renewals, _topology
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
)
FORMALISM_PASSES = (
    *_renewals.PASSES,
    *_lattices.PASSES,
    *_agents.PASSES,
    *_flows.PASSES,
    *_particles.PASSES,
    *_topology.PASSES,
    *_fields.PASSES,
    *_kinetics.PASSES,
)
STEPWISE_FORMS: dict[frozenset[type], Form] = {
    **_agents.STEPWISE_FORMS,
    **_lattices.STEPWISE_FORMS,
    **_renewals.STEPWISE_FORMS,
}
OWNER_RULES = (
    *_renewals.OWNER_RULES,
    *_lattices.OWNER_RULES,
    *_agents.OWNER_RULES,
    *_flows.OWNER_RULES,
    *_control.OWNER_RULES,
    *_topology.OWNER_RULES,
)


def register_formalisms() -> None:
    """Register the passes, forms, owner rules, and bindings of the formalisms that still compile here.

    Each formalism registers its own contributions once its passes move into its package.
    """
    for item in FORMALISM_PASSES:
        register_pass(item)
    for kinds, form in STEPWISE_FORMS.items():
        register_form(kinds, form, stepwise=True)
    for kind, companions in _renewals.COMBINED.items():
        register_combined(kind, companions)
    for rule in OWNER_RULES:
        register_owner(rule)
    for over, binding in _particles.BINDINGS.items():
        register_binding(over, binding)


register_formalisms()
