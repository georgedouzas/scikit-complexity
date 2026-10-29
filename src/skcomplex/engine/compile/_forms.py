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
from ...types import FormKind
from ._renewal_law import RenewalLaw

LATTICE_LAWS: tuple[type, ...] = (LocalRuleLaw, DriveLaw, RelaxationLaw, ExtremalLaw)
STEPWISE_FORMS: dict[frozenset[type], FormKind] = {
    frozenset({RenewalLaw}): FormKind.RENEWAL,
    frozenset({RuleLaw}): FormKind.AGENT,
    frozenset({SampleLaw}): FormKind.MONTE_CARLO,
    **{
        frozenset(kinds): FormKind.LATTICE
        for count in range(1, len(LATTICE_LAWS) + 1)
        for kinds in itertools.combinations(LATTICE_LAWS, count)
    },
}
"""Forms whose solvers evaluate kernels step by step on the NumPy backend, by the law classes that compile to them."""

_FORMS: dict[frozenset[type], FormKind] = {
    frozenset(): FormKind.ODE,
    frozenset({RateLaw}): FormKind.ODE,
    frozenset({NoiseLaw}): FormKind.SDE,
    frozenset({RateLaw, NoiseLaw}): FormKind.SDE,
    frozenset({JumpLaw}): FormKind.JUMP,
    frozenset({UpdateLaw}): FormKind.DISCRETE,
    **STEPWISE_FORMS,
}
"""Forms by the law classes that compile to them."""

_COMBINED: dict[type, frozenset[type]] = {RenewalLaw: frozenset({RateLaw, NoiseLaw})}
"""Law classes that combine with laws of other classes of the same components into their form, with those classes."""
