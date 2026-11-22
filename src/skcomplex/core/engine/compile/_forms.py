"""Name the forms of sets of law classes and infer the form of components."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools

from ...components.laws import BaseLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ...components.laws._local_rule import _LocalRuleLaw
from ...components.laws._rule import _RuleLaw
from ...components.laws._sample import _SampleLaw
from ...enums import FormKind
from ._array_update_law import ArrayUpdateLaw
from ._relaxation_law import RelaxationLaw
from ._renewal_law import RenewalLaw

LATTICE_LAWS: tuple[type, ...] = (_LocalRuleLaw, ArrayUpdateLaw, RelaxationLaw)
STEPWISE_FORMS: dict[frozenset[type[BaseLaw]], FormKind] = {
    frozenset({RenewalLaw}): FormKind.RENEWAL,
    frozenset({_RuleLaw}): FormKind.AGENT,
    frozenset({_SampleLaw}): FormKind.MONTE_CARLO,
    **{
        frozenset(kinds): FormKind.LATTICE
        for count in range(1, len(LATTICE_LAWS) + 1)
        for kinds in itertools.combinations(LATTICE_LAWS, count)
    },
}
FORMS: dict[frozenset[type[BaseLaw]], FormKind] = {
    frozenset(): FormKind.ODE,
    frozenset({RateLaw}): FormKind.ODE,
    frozenset({NoiseLaw}): FormKind.SDE,
    frozenset({RateLaw, NoiseLaw}): FormKind.SDE,
    frozenset({JumpLaw}): FormKind.JUMP,
    frozenset({UpdateLaw}): FormKind.DISCRETE,
    **STEPWISE_FORMS,
}
COMBINED: dict[type[BaseLaw], frozenset[type[BaseLaw]]] = {RenewalLaw: frozenset({RateLaw, NoiseLaw})}
