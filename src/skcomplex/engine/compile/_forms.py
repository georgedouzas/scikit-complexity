"""Name the forms that sets of law classes compile to, and infer the form of a set of components."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools

from ...core.laws import (
    DriveLaw,
    ExtremalLaw,
    JumpLaw,
    LocalRuleLaw,
    NoiseLaw,
    RateLaw,
    RelaxationLaw,
    RuleLaw,
    SampleLaw,
    UpdateLaw,
)
from ..problems import Form
from ._renewal_law import RenewalLaw

LATTICE_LAWS: tuple[type, ...] = (LocalRuleLaw, DriveLaw, RelaxationLaw, ExtremalLaw)
STEPWISE_FORMS: dict[frozenset[type], Form] = {
    frozenset({RenewalLaw}): Form.RENEWAL,
    frozenset({RuleLaw}): Form.AGENT,
    frozenset({SampleLaw}): Form.MONTE_CARLO,
    **{
        frozenset(kinds): Form.LATTICE
        for count in range(1, len(LATTICE_LAWS) + 1)
        for kinds in itertools.combinations(LATTICE_LAWS, count)
    },
}
"""Forms whose solvers evaluate kernels step by step on the NumPy backend, by the law classes that compile to them."""

_FORMS: dict[frozenset[type], Form] = {
    frozenset(): Form.ODE,
    frozenset({RateLaw}): Form.ODE,
    frozenset({NoiseLaw}): Form.SDE,
    frozenset({RateLaw, NoiseLaw}): Form.SDE,
    frozenset({JumpLaw}): Form.JUMP,
    frozenset({UpdateLaw}): Form.DISCRETE,
    **STEPWISE_FORMS,
}
"""Forms by the law classes that compile to them."""

_COMBINED: dict[type, frozenset[type]] = {RenewalLaw: frozenset({RateLaw, NoiseLaw})}
"""Law classes that combine with laws of other classes of the same components into their form, with those classes."""


