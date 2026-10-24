"""Declare a formalism's contributions to the core and register each with the concept that owns it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

from ..base import Form, Registry, RegistryError
from ..compile import (
    Backend,
    Binding,
    Check,
    OwnerRule,
    Pass,
    SystemPass,
    register_backend,
    register_binding,
    register_check,
    register_combined,
    register_form,
    register_owner,
    register_pass,
)
from ..components import Component
from ..laws import Law, register_laws
from ..solvers import Stepper, register_solver
from ..space import Space
from ..viz import Visual, register_visual


def _check_acyclic(formalism: 'Formalism') -> None:
    """Raise an error when the dependencies of a formalism, followed through the registered ones, lead back to it."""
    stack, seen = list(formalism.depends), set()
    while stack:
        name = stack.pop()
        if name == formalism.name:
            error_msg = f'The dependencies of the formalism {formalism.name!r} form a cycle through itself. Remove one.'
            raise RegistryError(error_msg)
        if name not in seen and name in FORMALISMS:
            seen.add(name)
            stack += FORMALISMS.resolve(name).depends


def _check_dependencies(formalism: 'Formalism') -> None:
    """Raise an error for dependencies that lead back to the formalism or are not registered."""
    _check_acyclic(formalism)
    missing = [name for name in formalism.depends if name not in FORMALISMS]
    if missing:
        error_msg = (
            f'The formalism {formalism.name!r} depends on {", ".join(map(repr, missing))}, which is not registered. '
            'Import the formalisms it depends on before registering it.'
        )
        raise RegistryError(error_msg)


def _is_registered(formalism: 'Formalism') -> bool:
    """Check whether a formalism is registered, raising an error when another formalism holds its name."""
    if formalism.name not in FORMALISMS:
        return False
    if FORMALISMS.resolve(formalism.name) == formalism:
        return True
    error_msg = f'Another formalism is registered as {formalism.name!r}. Give the formalism another name.'
    raise RegistryError(error_msg)


def _register_compilation(formalism: 'Formalism') -> None:
    """Register the passes, checks, stepwise forms, combined law classes, owner rules, and bindings of a formalism."""
    for system_pass in formalism.passes:
        register_pass(system_pass)
    for check in formalism.checks:
        register_check(check)
    for kinds, form in formalism.stepwise_forms.items():
        register_form(kinds, form, stepwise=True)
    for kind, companions in formalism.combined.items():
        register_combined(kind, companions)
    for rule in formalism.owners:
        register_owner(rule)
    for over, binding in formalism.bindings.items():
        register_binding(over, binding)


def register(formalism: 'Formalism') -> None:
    """Register a formalism and add each of its contributions to the registry of the concept that owns it.

    Law classes go to the laws, passes, checks, stepwise forms, combined law classes, owner rules, and bindings to
    compilation, solvers to the solvers of their form, backends to the backends, and visuals to visualization. Law
    kinds, traits, and forms are registered when they are created. Registering the same formalism again changes
    nothing.

    Args:
        formalism:
            The formalism.

    Raises:
        RegistryError:
            A dependency is not registered or leads back to the formalism, another formalism holds its name, or a
            law class misses a declaration.

    Examples:
        >>> from skcomplex.core.formalisms import Formalism, register
        >>> register(Formalism('dependent', depends=('missing',)))
        Traceback (most recent call last):
        ...
        skcomplex.core.base._errors.RegistryError: The formalism 'dependent' depends on 'missing', which is not ...
    """
    _check_dependencies(formalism)
    if _is_registered(formalism):
        return
    register_laws(*formalism.laws)
    _register_compilation(formalism)
    for name, stepper in formalism.solvers.items():
        register_solver(name, stepper, default=stepper in formalism.defaults)
    for backend in formalism.backends:
        register_backend(backend)
    for kind, visual in formalism.visuals.items():
        register_visual(kind, visual)
    FORMALISMS.register(formalism.name, formalism)


def registered_formalisms() -> tuple['Formalism', ...]:
    """Return the registered formalisms in registration order.

    Returns:
        formalisms:
            The registered formalisms.

    Examples:
        >>> from skcomplex.core.formalisms import registered_formalisms
        >>> all(formalism.name for formalism in registered_formalisms())
        True
    """
    return tuple(FORMALISMS.resolve(name) for name in FORMALISMS)


@dataclass(frozen=True)
class Formalism:
    """Declaration of everything a formalism contributes to the core, registered in one call to `register`.

    Args:
        name:
            The name of the formalism, the name of its package.

        depends:
            The names of the formalisms it builds on, which must be registered first.

        laws:
            The law classes it defines.

        components:
            The component classes it defines.

        spaces:
            The space classes it defines.

        passes:
            The passes that every compilation runs, ordered by the fields they read and write, and the passes that
            compile the whole systems they claim.

        forms:
            The forms it defines.

        solvers:
            The solver classes by the name users select them with.

        defaults:
            The solver classes, each also listed in `solvers`, that are the default of their form.

        backends:
            The backend classes.

        visuals:
            The visual classes by the name element types refer to them with.

        checks:
            The checks that validation runs, each taking a system and the census of its laws.

        stepwise_forms:
            The forms that run step by step, by the set of law classes whose components take them.

        combined:
            The law classes that combine with laws of other classes of the same components into their form, with
            those other classes.

        owners:
            The rules that name the owner of the laws they know.

        bindings:
            How the kernel spec of an interaction is built, by the binding it declares, such as `'pairs'`.

    Examples:
        >>> from skcomplex.core.formalisms import Formalism
        >>> Formalism('empty').depends
        ()
    """

    name: str
    depends: tuple[str, ...] = ()
    laws: tuple[type[Law], ...] = ()
    components: tuple[type[Component], ...] = ()
    spaces: tuple[type[Space], ...] = ()
    passes: tuple[Pass | SystemPass, ...] = ()
    forms: tuple[Form, ...] = ()
    solvers: Mapping[str, type[Stepper]] = field(default_factory=dict)
    defaults: tuple[type[Stepper], ...] = ()
    backends: tuple[type[Backend], ...] = ()
    visuals: Mapping[str, type[Visual]] = field(default_factory=dict)
    checks: tuple[Check, ...] = ()
    stepwise_forms: Mapping[frozenset[type], Form] = field(default_factory=dict)
    combined: Mapping[type, frozenset[type]] = field(default_factory=dict)
    owners: tuple[OwnerRule, ...] = ()
    bindings: Mapping[str, Binding] = field(default_factory=dict)


FORMALISMS = Registry[Formalism]('formalism')
