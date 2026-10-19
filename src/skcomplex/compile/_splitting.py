"""Split the rates of fields into a linear part with constant coefficients and a nonlinear remainder."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sage.all__sagemath_symbolics import SR

from ..core import Expression
from ..model import Field
from ._kernels import Reference, Source

OPERATORS = ('laplacian', 'partial', 'inverse_laplacian')

type Chain = tuple[tuple[str, int], ...]
type Terms = list[tuple[Expression, Chain, Expression]]


@dataclass(frozen=True, eq=False)
class LinearTerm:
    """Term of the rate of a field state that applies a chain of operators to a state, times a constant coefficient.

    Args:
        target:
            The component whose rate holds the term.

        source:
            The component the operators act on.

        place:
            The field of the source.

        chain:
            The operators from outermost to innermost, with their axes.

        coefficient:
            The coefficient, an expression of parameters.

        references:
            The references of the coefficient's symbols.
    """

    target: Reference
    source: Reference
    place: Field
    chain: Chain
    coefficient: Expression
    references: Mapping[Expression, Reference]


def _name_of(operator: object) -> str:
    """Return the name of a Python operator of a Sage expression, or an empty string."""
    return getattr(operator, '__name__', '')


def _operator_terms(name: str, operands: list[Any], states: Mapping[Expression, Any]) -> Terms | None:
    """Return the linear terms of an operator applied to a linear expression, or `None`."""
    inner = _linear(operands[0], states)
    if inner is None:
        return None
    axis = int(operands[1]) if name == 'partial' else 0
    return [(coefficient, ((name, axis), *chain), state) for coefficient, chain, state in inner]


def _sum_terms(operands: list[Any], states: Mapping[Expression, Any]) -> Terms | None:
    """Return the linear terms of a sum, or `None` when a summand is not linear."""
    terms: Terms = []
    for item in operands:
        part = _linear(item, states)
        if part is None:
            return None
        terms += part
    return terms


def _product_terms(operands: list[Any], states: Mapping[Expression, Any]) -> Terms | None:
    """Return the linear terms of a constant times a linear expression, or `None`."""
    constant, variable = SR(1), []
    for item in operands:
        if any(symbol in states for symbol in item.variables()):
            variable.append(item)
        else:
            constant *= item
    if len(variable) != 1:
        return None
    part = _linear(variable[0], states)
    return None if part is None else [(constant * coefficient, chain, state) for coefficient, chain, state in part]


def _linear(expression: Any, states: Mapping[Expression, Any]) -> Terms | None:  # noqa: ANN401
    """Return the linear terms of an expression in field states, or `None` when it is not linear."""
    expression = SR(expression)
    if not any(item in states for item in expression.variables()):
        return []
    if expression in states:
        return [(SR(1), (), expression)]
    operator = expression.operator()
    name = getattr(operator, 'name', lambda: '')()
    if name in OPERATORS:
        return _operator_terms(name, expression.operands(), states)
    combine = {'add_vararg': _sum_terms, 'mul_vararg': _product_terms}.get(_name_of(operator))
    return None if combine is None else combine(expression.operands(), states)


def linear_terms(
    value: object,
    target: Reference,
    references: Mapping[Expression, Reference],
    fields: Mapping[str, Field],
) -> list[LinearTerm]:
    """Return the linear terms with constant coefficients of the rate of a field state.

    Additive terms that are not linear in the field's states, or whose coefficients depend on states, are left out,
    and solvers treat them as the nonlinear remainder.

    Args:
        value:
            The rate as written, with its operators.

        target:
            The component the rate belongs to.

        references:
            The references of the symbols of the rate's component.

        fields:
            The field of every member entry by entry name.

    Returns:
        terms:
            The linear terms.
    """
    if not isinstance(value, Expression):
        return []
    states = {
        item: reference
        for item, reference in references.items()
        if reference.is_state and reference.source is Source.MEMBER and reference.entry.name in fields
    }
    expanded = SR(value).expand()
    parts = expanded.operands() if _name_of(expanded.operator()) == 'add_vararg' else [expanded]
    terms = []
    for part in parts:
        found = _linear(part, states)
        for coefficient, chain, state in found or []:
            if any(item in states or references.get(item, None) is None for item in coefficient.variables()):
                continue
            source = states[state]
            terms.append(
                LinearTerm(
                    target,
                    source,
                    fields[source.entry.name],
                    chain,
                    coefficient,
                    {item: references[item] for item in coefficient.variables()},
                ),
            )
    return terms
