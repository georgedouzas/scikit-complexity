"""Name the forms that sets of law classes compile to, and infer the form of a set of components."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Iterable

from ...core.errors import IncompatibleLawsError
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


def _fits_form(owners: Iterable[set[type]]) -> bool:
    """Check whether the law classes of components compile to one form.

    Args:
        owners:
            The law classes of every component.

    Returns:
        fits:
            Whether a form takes the law classes.
    """
    return _form_kinds(owners) in _FORMS


def _infer_form(owners: Iterable[set[type]]) -> Form:
    """Return the form that the law classes of the components of a system determine.

    A combined class, such as a renewal, drops out of the classes when the rest are its companions and no
    component has laws of that class alone.

    Args:
        owners:
            The law classes of every component.

    Returns:
        form:
            The form.

    Raises:
        IncompatibleLawsError:
            No form takes the law classes.
    """
    kinds = _form_kinds(owners)
    if kinds not in _FORMS:
        found = ', '.join(sorted(kind.__name__ for kind in kinds))
        supported = '; '.join(
            ', '.join(sorted(kind.__name__ for kind in combination)) for combination in _FORMS if combination
        )
        error_msg = f'The system mixes the law kinds {found}. Supported combinations are: {supported}.'
        raise IncompatibleLawsError(error_msg)
    return _FORMS[kinds]


def _is_stepwise(form: Form) -> bool:
    """Check whether the solvers of a form evaluate its kernels step by step on the NumPy backend.

    Args:
        form:
            The form.

    Returns:
        stepwise:
            Whether the form is stepwise.
    """
    return form in STEPWISE_FORMS.values()
