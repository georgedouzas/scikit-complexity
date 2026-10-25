"""Read derivatives and lags of states through their companion states, and solve equations for highest derivatives."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
from sage.all__sagemath_symbolics import SR, matrix

from ....core.base import (
    BaseLaw,
    Expression,
    SymbolicVector,
    Trait,
    ValidationError,
    VariableKind,
    components_of,
    declarations_of,
    has_trait,
    symbol,
    vector,
)
from ....core.laws import BoundaryLaw, Call, EquationLaw, EventLaw, JumpLaw, NoiseLaw, RateLaw, Symbols, UpdateLaw
from ....core.laws._laws import RenewalLaw
from .._compilation import BlockLaws, Compilation
from .._complex import split_law
from .._declarations import Block, Declarations, is_interaction
from .._lowering import calls_of
from .._pipeline import BasePass
from .._specs import KernelSpec, state_name

EQUATION_UNKNOWNS = 6


def _companion_symbol(spec: KernelSpec, item: Expression, suffix: str) -> Expression:
    """Return the symbol of a companion state of the state a symbol reads, in the same namespace and column."""
    reference = spec.references[item]
    name = f'{reference.entry.name}{suffix}'
    for other, candidate in spec.references.items():
        if (
            candidate.entry.name == name
            and candidate.column == reference.column
            and candidate.source == reference.source
        ):
            return other
    error_msg = f'A law of {spec.owner} reads {name}, which the namespace does not hold.'
    raise ValidationError(error_msg)


def _substitute(
    value: object,
    spec: KernelSpec,
    orders: _Orders,
    owner: str,
    highest: dict[Expression, Expression] | None = None,
) -> object:
    """Return a law value with derivatives and lags of states replaced by their companion symbols.

    When `highest` is given, derivatives of a state's own order become placeholders that map to the symbol of the
    state's highest derivative, which an equation solves for.
    """
    if isinstance(value, SymbolicVector):
        return vector(_substitute(item, spec, orders, owner, highest) for item in value)
    if not isinstance(value, Expression):
        return value
    replacements = {}
    for call in calls_of(value, 'derivative'):
        item, degree = call.operands()
        reference = spec.references.get(item)
        order = orders.orders.get(reference.entry.name, 1) if reference is not None else 1
        if highest is not None and reference is not None and degree.is_integer() and int(degree) == order:
            top = item if order == 1 else _companion_symbol(spec, item, f'_d{order - 1}')
            placeholder = symbol(f'_highest_{top}')
            highest[placeholder] = top
            replacements[call] = placeholder
            continue
        if reference is None or not degree.is_integer() or not 0 < int(degree) < order:
            error_msg = (
                f'{owner} uses {call}, but laws read derivatives of a state only below its declared order, here '
                f'{order}. Declare a higher order with `state(order=...)`, or state the highest derivative with an '
                '`EquationLaw`.'
            )
            raise ValidationError(error_msg)
        replacements[call] = _companion_symbol(spec, item, f'_d{int(degree)}')
    for call in calls_of(value, 'lag'):
        item, steps = call.operands()
        reference = spec.references.get(item)
        kept = orders.lags.get(reference.entry.name, 0) if reference is not None else 0
        if reference is None or not steps.is_integer() or not 0 < int(steps) <= kept:
            error_msg = (
                f'{owner} uses {call}, but the state keeps {kept} past values. Declare more with `state(lags=...)`.'
            )
            raise ValidationError(error_msg)
        replacements[call] = _companion_symbol(spec, item, f'_lag{int(steps)}')
    return value.subs(replacements) if replacements else value


def _retarget(law: RateLaw | NoiseLaw, spec: KernelSpec, orders: _Orders) -> RateLaw | NoiseLaw:
    """Return a rate or noise law that targets the highest derivative of its state, after checking its order."""
    items = components_of(law.target)
    references = [spec.references.get(item) for item in items]
    declared = {orders.orders.get(reference.entry.name, 1) for reference in references if reference is not None}
    order = max(declared, default=1)
    if law.order is not None and law.order != order:
        names = ', '.join(sorted({reference.entry.name for reference in references if reference is not None}))
        error_msg = (
            f'A {type(law).__name__} of {spec.owner} on {names} gives the derivative of order {law.order}, but the '
            f'state has order {order}. Match the law to the order declared with `state(order=...)`.'
        )
        raise ValidationError(error_msg)
    if order == 1:
        return law
    tops = [_companion_symbol(spec, item, f'_d{order - 1}') for item in items]
    target = vector(tops) if isinstance(law.target, SymbolicVector) else tops[0]
    return replace(law, target=target)


def _intrinsic_laws(spec: KernelSpec, block: Block, orders: _Orders) -> list[BaseLaw]:
    """Return the chain `x' = x_d1, ...` of every higher-order state and the shifts of every lagged state."""
    laws: list[BaseLaw] = []
    for variable, entry in block.entries.items():
        names = [variable, *(f'{variable}_d{index}' for index in range(1, orders.orders.get(entry.name, 1)))]
        laws += [
            RateLaw(getattr(spec.namespace, lower), getattr(spec.namespace, upper))
            for lower, upper in itertools.pairwise(names)
        ]
        shifted = [variable, *(f'{variable}_lag{index}' for index in range(1, orders.lags.get(entry.name, 0) + 1))]
        laws += [
            UpdateLaw(getattr(spec.namespace, older), getattr(spec.namespace, newer))
            for newer, older in itertools.pairwise(shifted)
        ]
    return laws


def _relation(spec: KernelSpec, law: EquationLaw, orders: _Orders) -> _Relation:
    """Return an equation law as a relation in placeholders of the highest derivatives it contains."""
    description = f'the equation {law.lhs} == {law.rhs} of {spec.owner}'
    if spec.pairs is not None:
        error_msg = f'Pair interaction {spec.owner} returns {description}. Pair interactions take only rate laws.'
        raise ValidationError(error_msg)
    highest: dict[Expression, Expression] = {}
    lhs = _substitute(SR(law.lhs), spec, orders, f'In {description}, it', highest)
    rhs = _substitute(SR(law.rhs), spec, orders, f'In {description}, it', highest)
    if not highest:
        error_msg = f'{description[0].upper()}{description[1:]} contains no highest derivative of a state to solve for.'
        raise ValidationError(error_msg)
    return _Relation(spec, SR(lhs) - SR(rhs), highest, description)


def _ordered_law(spec: KernelSpec, law: BaseLaw, orders: _Orders) -> object:
    """Return a jump, event, renewal, or assignment law with derivatives and lags substituted and rates retargeted."""
    owner = f'A {type(law).__name__} of {spec.owner}'
    if isinstance(law, JumpLaw):
        return JumpLaw(_substitute(law.rate, spec, orders, owner), law.changes)
    if isinstance(law, EventLaw | RenewalLaw):
        reset = {
            _substitute(SR(target), spec, orders, owner): _substitute(SR(value), spec, orders, owner)
            for target, value in (law.reset or {}).items()
        }
        reset_value = reset if law.reset is not None else None
        if isinstance(law, RenewalLaw):
            return replace(law, waiting=_substitute(SR(law.waiting), spec, orders, owner), reset=reset_value)
        return replace(law, guard=_substitute(SR(law.guard), spec, orders, owner), reset=reset_value)
    if not isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw):
        return law
    value = _substitute(law.value, spec, orders, owner)
    if isinstance(law, RateLaw | NoiseLaw):
        return _retarget(replace(law, value=value), spec, orders)
    return type(law)(law.target, value)


def _with_orders(spec: KernelSpec, laws: list[BaseLaw], block: Block, orders: _Orders) -> list[Any]:
    """Return laws with derivatives and lags substituted, rates retargeted, equations as relations, and chains added."""
    if not orders.orders and not orders.lags and not any(isinstance(law, EquationLaw) for law in laws):
        return laws
    result: list[Any] = []
    for law in laws:
        if has_trait(law, Trait.KERNEL_FREE):
            result.append(law)
        elif isinstance(law, EquationLaw):
            result.append(_relation(spec, law, orders))
        else:
            result.append(_ordered_law(spec, law, orders))
    if not is_interaction(block.node):
        result += [
            item
            for law in _intrinsic_laws(spec, block, orders)
            for item in split_law(law, spec.owner, lambda target: state_name(spec, target))
        ]
    return result


def _numeric_solution(rows: int, size: int, index: int) -> Callable[..., Any]:
    """Return a function of the matrix and right side entries that solves the linear system for one unknown."""

    def solve(t: float, *entries: Any) -> Any:  # noqa: ANN401
        del t
        values = [np.broadcast_to(np.asarray(entry, dtype=float), (rows,)) for entry in entries]
        matrix_ = np.stack(values[: size * size], axis=-1).reshape(rows, size, size)
        right = np.stack(values[size * size :], axis=-1)
        return np.linalg.solve(matrix_, right[..., None])[..., 0][:, index]

    return solve


def _solve_equations(specs: list[tuple[KernelSpec, list[Any]]]) -> list[tuple[KernelSpec, list[BaseLaw]]]:
    """Return specs whose equations are replaced by rate laws on the highest derivatives they solve for jointly."""
    relations = [item for _, laws in specs for item in laws if isinstance(item, _Relation)]
    kept = [(spec, [law for law in laws if not isinstance(law, _Relation)]) for spec, laws in specs]
    if not relations:
        return kept
    highest: dict[Expression, Expression] = {}
    for relation in relations:
        highest.update(relation.highest)
    unknowns = list(highest)
    for relation in relations:
        for first in unknowns:
            for second in unknowns:
                if not relation.expression.diff(first).diff(second).is_trivial_zero():
                    error_msg = (
                        f'{relation.description[0].upper()}{relation.description[1:]} is not linear in the highest '
                        f'derivative {highest[first]}. Write the equation so that highest derivatives appear linearly.'
                    )
                    raise ValidationError(error_msg)
    descriptions = '; '.join(relation.description for relation in relations)
    if len(relations) != len(unknowns):
        error_msg = (
            f'{len(relations)} equations give {len(unknowns)} highest derivatives, '
            f'{", ".join(str(highest[item]) for item in unknowns)}, in {descriptions}. Give one equation per '
            'derivative.'
        )
        raise ValidationError(error_msg)
    zero = dict.fromkeys(unknowns, 0)
    coefficients = [[relation.expression.diff(item) for item in unknowns] for relation in relations]
    right = [-relation.expression.subs(zero) for relation in relations]
    references = {key: value for relation in relations for key, value in relation.spec.references.items()}
    rows = max(relation.spec.rows for relation in relations)
    owner = ', '.join(dict.fromkeys(relation.spec.owner for relation in relations))
    merged = KernelSpec(owner, Symbols(), references, rows)
    laws: list[BaseLaw]
    if len(unknowns) <= EQUATION_UNKNOWNS:
        system = matrix(SR, coefficients)
        if system.det().simplify_full().is_trivial_zero():
            error_msg = f'The highest derivatives of {descriptions} cannot be solved for: the system is singular.'
            raise ValidationError(error_msg)
        solution = system.solve_right(vector([SR(item) for item in right]), check=False)
        laws = [RateLaw(highest[item], value) for item, value in zip(unknowns, solution, strict=True)]
    else:
        entries = [entry for row in coefficients for entry in row] + right
        laws = [
            RateLaw(highest[item], Call(_numeric_solution(rows, len(unknowns), index), *entries))
            for index, item in enumerate(unknowns)
        ]
    return [*kept, (merged, laws)]


@dataclass(frozen=True)
class _Orders:
    """Orders and lags of the states of a system by entry name."""

    orders: dict[str, int]
    lags: dict[str, int]

    @classmethod
    def of(cls, blocks: list[Block]) -> _Orders:
        """Return the orders above 1 and the lags of every state of the blocks."""
        orders, lags = {}, {}
        for block in blocks:
            if is_interaction(block.node) or block.node.is_agent:
                continue
            declarations = declarations_of(block.node.component_type, VariableKind.STATE)
            for variable, entry in block.entries.items():
                declaration = declarations.get(variable)
                if declaration is not None and declaration.derivatives:
                    orders[entry.name] = declaration.derivatives + 1
                if declaration is not None and declaration.lags:
                    lags[entry.name] = declaration.lags
        return cls(orders, lags)


@dataclass(frozen=True)
class _Relation:
    """Equation of a component as `expression == 0`, with the placeholders of its highest derivatives."""

    spec: KernelSpec
    expression: Expression
    highest: dict[Expression, Expression]
    description: str


class OrdersPass(BasePass):
    """Pass that reads derivatives and lags through companion states, adds their chains, and solves equations."""

    reads = ('lowered', 'layout')
    writes = ('ordered',)

    def _run(self, compilation: Compilation) -> None:
        """Lower the orders and lags of the laws and solve their equations.

        Args:
            compilation:
                The compilation state.
        """
        blocks = list(cast('Declarations', compilation.layout).blocks.values())
        orders = _Orders.of(blocks)
        compilation.ordered = _solve_equations(
            [
                (spec, _with_orders(spec, laws, block, orders))
                for spec, laws, block in cast('BlockLaws', compilation.lowered)
            ],
        )
