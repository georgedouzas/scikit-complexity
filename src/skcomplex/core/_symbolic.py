"""Give every layer the Sage symbolic ring, its symbols, and its vectors."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterable
from typing import Any

from sage.all__sagemath_symbolics import SR
from sage.all__sagemath_symbolics import vector as sage_vector
from sage.modules.free_module_element import FreeModuleElement as SymbolicVector
from sage.symbolic.assumptions import assume
from sage.symbolic.expression import Expression

from ._domains import AnyDomain, Categorical, Domain

_DOMAINS: dict[str, AnyDomain] = {}
_DECLARED: set[str] = set()


def symbol(name: str, domain: AnyDomain = Domain.REAL) -> Expression:
    """Return the Sage symbol of a name, real or integer, and record its variable domain.

    The symbol is declared real or integer without Maxima, which Sage would otherwise load to record the domain, so
    compiling stays fast. `declare_domains` gives Maxima the full domains before an exact solution.

    Args:
        name:
            The name of the symbol.

        domain:
            The domain of the variable the symbol stands for.

    Returns:
        symbol:
            The Sage symbol.

    Examples:
        >>> from skcomplex.core import Domain, domain_of, symbol
        >>> n = symbol('n', Domain.POSITIVE)
        >>> n.is_real(), domain_of(n)
        (True, <Domain.POSITIVE: 'positive'>)
    """
    item = SR.symbol(name)
    item.decl_assume('integer' if domain.sage_domain == 'integer' else 'real')
    _DOMAINS[name] = domain
    return item


def vector(components: Iterable[Any]) -> SymbolicVector:
    """Return a Sage vector of symbolic components.

    Args:
        components:
            The components.

    Returns:
        vector:
            The vector over the symbolic ring.

    Examples:
        >>> from skcomplex.core import symbol, vector
        >>> x, y = symbol('x'), symbol('y')
        >>> vector([x, y]).dot_product(vector([x, y]))
        x^2 + y^2
    """
    return sage_vector(SR, list(components))


def components_of(value: Any) -> list[Any]:  # noqa: ANN401
    """Return the components of a vector, or a one-element list of any other value.

    Args:
        value:
            A vector, an expression, or a number.

    Returns:
        components:
            The components.
    """
    return list(value) if isinstance(value, SymbolicVector) else [value]


def symbols_of(expressions: Iterable[Expression]) -> set[Expression]:
    """Return the symbols that occur in expressions.

    Args:
        expressions:
            The expressions.

    Returns:
        symbols:
            The symbols.
    """
    return {item for expression in expressions for item in SR(expression).variables()}


def domain_of(item: Expression) -> AnyDomain:
    """Return the variable domain of a symbol.

    Args:
        item:
            The symbol.

    Returns:
        domain:
            The domain the symbol was created with, or `REAL` for a symbol the framework did not create.
    """
    return _DOMAINS.get(str(item), Domain.REAL)


def declare_domains(items: Iterable[Expression]) -> None:
    """Give Maxima the domains of symbols, once per symbol, so exact solutions use them.

    Args:
        items:
            The symbols.

    Raises:
        ValueError:
            A domain contradicts an assumption Maxima already holds.
    """
    for item in items:
        if str(item) in _DECLARED:
            continue
        domain = domain_of(item)
        if isinstance(domain, Categorical) or domain in {Domain.INTEGER, Domain.NONNEGATIVE_INTEGER, Domain.BOOLEAN}:
            assume(item, 'integer')
        elif domain in {Domain.REAL, Domain.ANGLE}:
            assume(item, 'real')
        if domain is Domain.POSITIVE:
            assume(item > 0)
        elif isinstance(domain, Categorical) or domain in {
            Domain.NONNEGATIVE,
            Domain.NONNEGATIVE_INTEGER,
            Domain.BOOLEAN,
        }:
            assume(item >= 0)
        _DECLARED.add(str(item))
