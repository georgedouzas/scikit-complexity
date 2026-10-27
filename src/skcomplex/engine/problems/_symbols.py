"""Create the Sage symbols that compiled laws are written in, and read the symbols and components of expressions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterable

from sage.all__sagemath_symbolics import CC, RR, SR, ZZ, Expression
from sage.all__sagemath_symbolics import vector as sage_vector
from sage.modules.free_module_element import FreeModuleElement

from ._domains import sage_assumption


def symbol(name: str, domain: object = RR) -> Expression:
    """Return the Sage symbol of a name, declared real or integer, or left complex, for the values of a Sage set.

    The symbol is declared without Maxima, which Sage would otherwise load to record the domain, so compiling stays
    fast. Sage keeps one symbol per name, so a name declared before with another domain is declared again.

    Args:
        name:
            The name of the symbol.

        domain:
            The Sage set of the values the symbol stands for.

    Returns:
        symbol:
            The Sage symbol.

    Examples:
        >>> from sage.all__sagemath_symbolics import CC
        >>> from skcomplex.engine.problems import symbol
        >>> symbol('z').is_real(), symbol('z', CC).is_real()
        (True, False)
    """
    item = SR.symbol(name)
    for previous in ('integer', 'real'):
        item.decl_forget(previous)
    assumption = sage_assumption(domain)
    if assumption != 'complex':
        item.decl_assume('integer' if assumption == 'integer' else 'real')
    return item


def domain_of(item: Expression) -> object:
    """Return the Sage set a symbol was declared for: the integers, the reals, or the complex numbers.

    Args:
        item:
            The symbol.

    Returns:
        domain:
            `ZZ`, `RR`, or `CC`.

    Examples:
        >>> from sage.all__sagemath_symbolics import ZZ
        >>> from skcomplex.engine.problems import domain_of, symbol
        >>> domain_of(symbol('k', ZZ))
        Integer Ring
    """
    if item.is_integer():
        return ZZ
    return RR if item.is_real() else CC


def vector(components: Iterable[object]) -> FreeModuleElement:
    """Return a vector over the symbolic ring.

    Args:
        components:
            The components.

    Returns:
        vector:
            The vector.
    """
    return sage_vector(SR, list(components))


def components_of(value: object) -> list[object]:
    """Return the components of a vector, or a scalar as its only component.

    Args:
        value:
            A vector, an expression, or a number.

    Returns:
        components:
            The components.
    """
    return list(value) if isinstance(value, FreeModuleElement) else [value]


def symbols_of(expressions: Iterable[object]) -> set[Expression]:
    """Return the symbols that expressions read.

    Args:
        expressions:
            The expressions.

    Returns:
        symbols:
            The symbols.
    """
    return {item for expression in expressions for item in SR(expression).variables()}
