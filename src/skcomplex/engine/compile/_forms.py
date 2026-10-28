"""Name the forms that sets of law classes compile to, and infer the form of a set of components."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Iterable

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
from ._laws import _FORMLESS_LAWS
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


def _form_kinds(owners: Iterable[set[type]]) -> frozenset[type]:
    """Return the law classes that decide the form of components with their own law classes.

    A combined class, such as a renewal, drops out of the classes when the rest are its companions and no
    component has laws of that class alone. A component with laws of that class alone next to components with other laws
    keeps it, so the system does not fit one form and runs as partitions.
    """
    owned = [{kind for kind in kinds if not issubclass(kind, _FORMLESS_LAWS)} for kinds in owners]
    kinds: set[type] = set().union(*owned)
    for kind, companions in _COMBINED.items():
        rest = kinds - {kind}
        if kind not in kinds or not rest:
            continue
        alone = any(item == {kind} for item in owned)
        if not alone and rest <= companions:
            kinds = rest
    return frozenset(kinds)


