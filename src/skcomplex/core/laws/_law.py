"""Define the base of every law, which declares its kind, its time class, and its traits, and the registry of laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass
from typing import ClassVar

from ..base import LawClass, LawKind, Registry, RegistryError, Requires, Trait


def _check_declarations(cls: type) -> None:
    """Raise an error for a class that is not a law class declaring its kind, its time class, and its requirements."""
    if not (isinstance(cls, type) and issubclass(cls, Law)):
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


def register_laws(*classes: type['Law']) -> None:
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
        >>> from skcomplex.core.laws import Rate, law_classes, register_laws
        >>> register_laws(Rate)
        >>> law_classes().count(Rate)
        1
    """
    for cls in classes:
        _check_declarations(cls)
    for cls in classes:
        LAWS.register(_qualified_name(cls), cls)


def law_classes(trait: Trait | None = None) -> tuple[type['Law'], ...]:
    """Return every registered law class, optionally only the classes with a trait, in registration order.

    Args:
        trait:
            The trait the classes declare, or `None` for every class.

    Returns:
        classes:
            The law classes.

    Examples:
        >>> from skcomplex.core.base import Trait
        >>> from skcomplex.core.formalisms.fields import Flux
        >>> from skcomplex.core.laws import law_classes
        >>> Flux in law_classes(Trait.FORMLESS)
        True
    """
    classes = [LAWS.resolve(key) for key in LAWS]
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
        >>> from skcomplex.core.laws import Rate, is_registered
        >>> is_registered(Rate(symbol('x'), 1)), is_registered(object())
        (True, False)
    """
    cls = law if isinstance(law, type) else type(law)
    name = _qualified_name(cls)
    return name in LAWS and LAWS.resolve(name) is cls


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
        >>> from skcomplex.core.laws import Boundary, Rate, has_trait
        >>> x = symbol('x')
        >>> has_trait(Boundary(x, 0), Trait.FORMLESS), has_trait(Rate(x, 1), Trait.FORMLESS)
        (True, False)
    """
    cls = law if isinstance(law, type) else type(law)
    return issubclass(cls, Law) and trait in cls.traits


@dataclass(frozen=True)
class Law:
    """Base of every law, a frozen dataclass that declares its kind, its time class, and its traits.

    A law class declares as class variables its `law_kind`, its `time_class` among continuous, event, discrete,
    constraint, and any, the `traits` the compiler and validation read, and what it `requires` of the system. A system
    holds only laws of registered classes, which the shared core and the formalisms register.

    Examples:
        >>> from skcomplex.core.base import LawClass, LawKind
        >>> from skcomplex.core.laws import Law, Rate
        >>> issubclass(Rate, Law), Rate.law_kind is LawKind.RATE, Rate.time_class is LawClass.CONTINUOUS
        (True, True, True)
    """

    law_kind: ClassVar[LawKind]
    time_class: ClassVar[LawClass]
    traits: ClassVar[frozenset[Trait]] = frozenset()
    requires: ClassVar[Requires] = Requires()


LAWS = Registry[type[Law]]('law')
