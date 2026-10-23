"""Define the base of every law, which declares its kind, its time class, and its traits."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass
from typing import ClassVar

from ..base import LawClass, LawKind, Requires, Trait


def law_classes(trait: Trait | None = None) -> tuple[type['Law'], ...]:
    """Return every law class that declares a kind, optionally only the classes with a trait.

    The law classes are the subclasses of `Law` of the imported modules, in definition order.

    Args:
        trait:
            The trait the classes declare, or `None` for every class.

    Returns:
        classes:
            The law classes.

    Examples:
        >>> from skcomplex.core.base import Trait
        >>> from skcomplex.core.laws import Flux, law_classes
        >>> Flux in law_classes(Trait.FORMLESS)
        True
    """
    found: list[type[Law]] = []
    stack = list(reversed(Law.__subclasses__()))
    while stack:
        cls = stack.pop()
        if cls not in found:
            found.append(cls)
        stack += reversed(cls.__subclasses__())
    declared = [cls for cls in found if hasattr(cls, 'law_kind')]
    return tuple(cls for cls in declared if trait is None or trait in cls.traits)


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
    constraint, and any, the `traits` the compiler and validation read, and what it `requires` of the system.

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
