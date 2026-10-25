"""Define the assumptions Maxima holds on the domains and signs of symbols, for exact solutions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterable, Mapping

from sage.symbolic.assumptions import assume, forget

from ...core.base import CategoricalDomain, Expression, NumericDomain, domain_of

DECLARED: set[str] = set()
SIGNS: dict[str, Expression] = {}


def declare_domains(items: Iterable[Expression], values: Mapping[Expression, float] | None = None) -> None:
    """Give Maxima the domains of symbols, once per symbol, so exact solutions use them.

    A symbol of a real domain with a nonzero value also gets the sign of its value, which replaces the sign of an
    earlier value.

    Args:
        items:
            The symbols.

        values:
            Current values of symbols whose sign Maxima takes when their domain is real.

    Raises:
        ValueError:
            A domain contradicts an assumption Maxima already holds.
    """
    for item in items:
        name = str(item)
        domain = domain_of(item)
        if name not in DECLARED:
            if isinstance(domain, CategoricalDomain) or domain in {
                NumericDomain.INTEGER,
                NumericDomain.NONNEGATIVE_INTEGER,
                NumericDomain.BOOLEAN,
            }:
                assume(item, 'integer')
            elif domain in {NumericDomain.REAL, NumericDomain.ANGLE}:
                assume(item, 'real')
            if domain is NumericDomain.POSITIVE:
                assume(item > 0)
            elif isinstance(domain, CategoricalDomain) or domain in {
                NumericDomain.NONNEGATIVE,
                NumericDomain.NONNEGATIVE_INTEGER,
                NumericDomain.BOOLEAN,
            }:
                assume(item >= 0)
            DECLARED.add(name)
        value = (values or {}).get(item)
        if domain is not NumericDomain.REAL or not value:
            continue
        sign = item > 0 if value > 0 else item < 0
        previous = SIGNS.get(name)
        if previous is not None and str(previous) == str(sign):
            continue
        if previous is not None:
            forget(previous)
        assume(sign)
        SIGNS[name] = sign
