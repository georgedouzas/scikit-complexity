"""Register the forms that sets of law classes compile to, and infer the form of a set of components."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterable, Mapping
from typing import cast

from ..base import Form, IncompatibleLawsError, Registry, Trait
from ..laws import Jump, Noise, Rate, Update, has_trait

CORE_FORMS: Mapping[frozenset[type], Form] = {
    frozenset(): Form.ODE,
    frozenset({Rate}): Form.ODE,
    frozenset({Noise}): Form.SDE,
    frozenset({Rate, Noise}): Form.SDE,
    frozenset({Jump}): Form.JUMP,
    frozenset({Update}): Form.DISCRETE,
}

FORMS = Registry[Form]('form')
STEPWISE = Registry[Form]('stepwise form')
COMBINED = Registry[frozenset[type]]('combined law')


def _forms() -> dict[frozenset[type], Form]:
    """Return the forms of the shared core followed by the registered forms."""
    return {**CORE_FORMS, **{cast('frozenset[type]', key): FORMS.resolve(key) for key in FORMS}}


def _form_kinds(owners: Iterable[set[type]]) -> frozenset[type]:
    """Return the law classes that decide the form of components with their own law classes.

    A registered combined class, such as a renewal, drops out of the classes when the rest are its companions and no
    component has laws of that class alone. A component with laws of that class alone next to components with other laws
    keeps it, so the system does not fit one form and runs as partitions.
    """
    owned = [{kind for kind in kinds if not has_trait(kind, Trait.FORMLESS)} for kinds in owners]
    kinds: set[type] = set().union(*owned)
    for key in COMBINED:
        kind = cast('type', key)
        rest = kinds - {kind}
        if kind not in kinds or not rest:
            continue
        alone = any(item == {kind} for item in owned)
        if not alone and rest <= COMBINED.resolve(key):
            kinds = rest
    return frozenset(kinds)


def fits_form(owners: Iterable[set[type]]) -> bool:
    """Check whether the law classes of components compile to one registered form.

    Args:
        owners:
            The law classes of every component.

    Returns:
        fits:
            Whether a form takes the law classes.
    """
    return _form_kinds(owners) in _forms()


def infer_form(owners: Iterable[set[type]]) -> Form:
    """Return the form that the law classes of the components of a system determine.

    A registered combined class, such as a renewal, drops out of the classes when the rest are its companions and no
    component has laws of that class alone.

    Args:
        owners:
            The law classes of every component.

    Returns:
        form:
            The form.

    Raises:
        IncompatibleLawsError:
            No registered form takes the law classes.
    """
    kinds = _form_kinds(owners)
    forms = _forms()
    if kinds not in forms:
        found = ', '.join(sorted(kind.__name__ for kind in kinds))
        supported = '; '.join(
            ', '.join(sorted(kind.__name__ for kind in combination)) for combination in forms if combination
        )
        error_msg = f'The system mixes the law kinds {found}. Supported combinations are: {supported}.'
        raise IncompatibleLawsError(error_msg)
    return forms[kinds]


def register_form(kinds: frozenset[type], form: Form, *, stepwise: bool = False) -> None:
    """Register the form of a problem whose components have laws of a set of classes.

    Args:
        kinds:
            The law classes, without the laws that decide no form.

        form:
            The form.

        stepwise:
            Whether the solvers of the form evaluate the kernels step by step on the NumPy backend, whatever the backend
            asked for.
    """
    FORMS.register(kinds, form)
    if stepwise:
        STEPWISE.register(form, form)


def register_combined(kind: type, companions: frozenset[type]) -> None:
    """Register a law class that combines with laws of other classes of the same components into their form.

    Args:
        kind:
            The law class, such as a renewal, which drops out of the classes that decide the form when the components
            of a system with laws of the class have laws of the companion classes alone besides.

        companions:
            The law classes it combines with.
    """
    COMBINED.register(kind, companions)


def is_stepwise(form: Form) -> bool:
    """Check whether the solvers of a form evaluate its kernels step by step on the NumPy backend.

    Args:
        form:
            The form.

    Returns:
        stepwise:
            Whether the form was registered as stepwise.
    """
    return form in set(STEPWISE)
