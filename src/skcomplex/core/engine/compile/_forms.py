"""Name the forms of sets of law classes and infer the form of components."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools

from ...components.laws import (
    ArrayUpdateLaw,
    CascadeLaw,
    ExtremalLaw,
    JumpLaw,
    LocalRuleLaw,
    NoiseLaw,
    RateLaw,
    RuleLaw,
    SampleLaw,
    UpdateLaw,
)
from ...types import FormKind
from ._renewal_law import RenewalLaw

LATTICE_LAWS: tuple[type, ...] = (LocalRuleLaw, ArrayUpdateLaw, CascadeLaw, ExtremalLaw)
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
FORMS: dict[frozenset[type], FormKind] = {
    frozenset(): FormKind.ODE,
    frozenset({RateLaw}): FormKind.ODE,
    frozenset({NoiseLaw}): FormKind.SDE,
    frozenset({RateLaw, NoiseLaw}): FormKind.SDE,
    frozenset({JumpLaw}): FormKind.JUMP,
    frozenset({UpdateLaw}): FormKind.DISCRETE,
    **STEPWISE_FORMS,
}
COMBINED: dict[type, frozenset[type]] = {RenewalLaw: frozenset({RateLaw, NoiseLaw})}
