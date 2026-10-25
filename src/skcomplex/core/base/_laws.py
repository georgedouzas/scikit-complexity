"""Define the base of the laws, the rules by which the states of a system change, and their registry."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Self

from ._errors import RegistryError
from ._kinds import LawClass, LawKind, Requires, Trait
from ._utils import Registry


def _check_declarations(cls: type) -> None:
    """Raise an error for a class that is not a law class declaring its kind, its time class, and its requirements."""
    if not (isinstance(cls, type) and issubclass(cls, BaseLaw)):
        error_msg = f'{cls!r} is not a law class. Register subclasses of `Law` only.'
        raise RegistryError(error_msg)
    missing = [
        name
        for name, term in (('law_kind', LawKind), ('time_class', LawClass), ('requires', Requires))
        if not isinstance(getattr(cls, name, None), term)
    ]
    if missing:
        error_msg = f'The law class {cls.__name__} declares no {", ".join(missing)}. Declare each as a class variable.'
        raise RegistryError(error_msg)


def _qualified_name(cls: type) -> str:
    """Return the module and the qualified name of a class, the key it is registered under."""
    return f'{cls.__module__}.{cls.__qualname__}'


def register_laws(*classes: type['BaseLaw']) -> None:
    """Register law classes, so systems may hold their laws and the compiler and validation read their declarations.

    The shared core registers its own law classes, and a formalism registers its law classes through its declaration.
    Registering a class again keeps it once.

    Args:
        *classes:
            The law classes.

    Raises:
        RegistryError:
            A class is not a law class, or does not declare its kind, its time class, or its requirements.

    Examples:
        >>> from skcomplex.core.laws import RateLaw
        >>> from skcomplex.core.base import law_classes, register_laws
        >>> register_laws(RateLaw)
        >>> law_classes().count(RateLaw)
        1
    """
    for cls in classes:
        _check_declarations(cls)
    for cls in classes:
        LAWS.register(_qualified_name(cls), cls)


def law_classes(trait: Trait | None = None) -> tuple[type['BaseLaw'], ...]:
    """Return every registered law class, optionally only the classes with a trait, in registration order.

    Args:
        trait:
            The trait the classes declare, or `None` for every class.

    Returns:
        classes:
            The law classes.

    Examples:
        >>> from skcomplex.core.base import Trait
        >>> from skcomplex.formalisms.fields import FluxLaw
        >>> from skcomplex.core.base import law_classes
        >>> FluxLaw in law_classes(Trait.FORMLESS)
        True
    """
    classes = [LAWS.resolve(key) for key in LAWS.keys]
    return tuple(cls for cls in classes if trait is None or trait in cls.traits)


def is_registered(law: object) -> bool:
    """Check whether a law or a law class is of a registered law class.

    Args:
        law:
            The law or the law class.

    Returns:
        registered:
            Whether its class is registered.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import RateLaw
        >>> from skcomplex.core.base import is_registered
        >>> is_registered(RateLaw(symbol('x'), 1)), is_registered(object())
        (True, False)
    """
    cls = law if isinstance(law, type) else type(law)
    name = _qualified_name(cls)
    return name in LAWS.keys and LAWS.resolve(name) is cls


def has_trait(law: object, trait: Trait) -> bool:
    """Check whether a value is a law or a law class that declares a trait.

    Args:
        law:
            The value, a law, a law class, or any other object.

        trait:
            The trait.

    Returns:
        has_trait:
            Whether the value is a law or a law class with the trait.

    Examples:
        >>> from skcomplex.core.base import Trait, symbol
        >>> from skcomplex.core.laws import BoundaryLaw, RateLaw
        >>> from skcomplex.core.base import has_trait
        >>> x = symbol('x')
        >>> has_trait(BoundaryLaw(x, 0), Trait.FORMLESS), has_trait(RateLaw(x, 1), Trait.FORMLESS)
        (True, False)
    """
    cls = law if isinstance(law, type) else type(law)
    return issubclass(cls, BaseLaw) and trait in cls.traits


@dataclass(frozen=True)
class BaseLaw:
    """Base of every law, a frozen dataclass of the states it changes, with its kind, its time class, and its traits.

    A law class declares as class variables its `law_kind`, its `time_class` among continuous, event, discrete,
    constraint, and any, the `traits` the compiler and validation read, and what it `requires` of the system. A system
    holds only laws of registered classes, which the shared core and the formalisms register.

    Every law has the field `targets`, the states it changes. A law that changes given states redeclares it as its
    first field, which takes one state or a tuple of states, and checking turns it into a tuple. Other laws leave it
    empty, or set it from their other values when they are checked.

    Args:
        targets:
            The states the law changes, one state or a tuple of them, a tuple once the law is checked.

    Examples:
        >>> from skcomplex.core.base import LawClass, LawKind
        >>> from skcomplex.core.laws import RateLaw
        >>> from skcomplex.core.base import BaseLaw
        >>> issubclass(RateLaw, BaseLaw), RateLaw.law_kind == LawKind.RATE, RateLaw.time_class is LawClass.CONTINUOUS
        (True, True, True)
        >>> from skcomplex.core.base import symbol
        >>> x = symbol('x')
        >>> RateLaw(x, -x).check().targets
        (x,)
    """

    law_kind: ClassVar[LawKind]
    time_class: ClassVar[LawClass]
    traits: ClassVar[frozenset[Trait]] = frozenset()
    requires: ClassVar[Requires] = Requires()

    targets: tuple[Any, ...] = field(default_factory=tuple, kw_only=True)

    def _check(self) -> Self:
        """Return the law with its values checked and normalized, the law itself by default."""
        return self

    def check(self) -> Self:
        """Return the law with its values checked and normalized.

        Laws are values, so checking returns a law rather than changing this one. Validation checks every law of a
        system before compiling it.

        Returns:
            law:
                The checked law, with its targets as a tuple and its values as Sage expressions.

        Raises:
            ValidationError:
                A value of the law is invalid, such as a NumPy array or a SymPy expression.
        """
        targets = self.targets
        if not isinstance(targets, tuple):
            targets = tuple(targets) if isinstance(targets, list) else (targets,)
            return replace(self, targets=targets)._check()
        return self._check()


LAWS = Registry[type[BaseLaw]]('law')
