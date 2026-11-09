"""Read derivatives and lags of states and solve for highest derivatives.

Derivatives and lags are read through companion states, and equations are solved for highest derivatives.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import itertools
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, matrix, solve
from sage.modules.free_module_element import FreeModuleElement

from ....components.interactions import ConnectionInteraction
from ....components.laws import BaseLaw, BoundaryLaw, CrossingLaw, EquationLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....components.laws.utils._reductions import reductions_in
from ....components.spaces import StaticTime
from ....errors import ValidationError
from ....specs import CallSpec, IssueSpec
from ....specs._orders_spec import _OrdersSpec
from ....types import Array, VariableKind
from ..._utils import components_of, declarations_of, symbol, vector
from ...problems._block_spec import BlockSpec
from ...problems._kernel_spec import KernelSpec
from ...problems._relation_spec import _RelationSpec
from .._base import BasePass
from .._compilation import Compilation
from .._laws import KERNEL_FREE_LAWS
from .._renewal_law import RenewalLaw
from ._complex import LowerComplexPass
from ._equation_blocks import EquationBlocksLowering

EQUATION_UNKNOWNS = 6


class OrdersPass(BasePass):
    """Pass that reads derivatives and lags through companion states.

    It also adds their chains and solves equations.
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('lowered', 'layout', 'programs')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('ordered',)

    def _run(self, inputs: object = None) -> None:
        """Lower the orders and lags of the laws and solve their equations."""
        compilation = self.compilation_
        blocks = list(Compilation._get_required(compilation.layout, 'layout').blocks.values())
        orders = self._get_orders_of(blocks, self._get_program_variables(compilation))
        specs = self._get_specs_of(compilation, orders)
        if isinstance(compilation.system.time, StaticTime):
            compilation.ordered = self._solve_static(specs)
        elif orders.algebraic:
            relations, unknowns = self._get_unknowns_of(specs, blocks, orders)
            lowering = EquationBlocksLowering(relations, unknowns).prepare(compilation)
            compilation.ordered = lowering.run(specs).lowered_
        else:
            compilation.ordered = self._solve_equations(specs)

    @staticmethod
    def _get_connected_ports(compilation: Compilation) -> set[tuple[str, str]]:
        """Return the element path and port name of every port a connection names."""
        return {
            (f'{node.scope}{member}', port)
            for node in Compilation._get_required(compilation.nodes, 'nodes')
            if isinstance(node.component, ConnectionInteraction)
            for member, port in zip(node.component.members, node.component.ports, strict=True)
        }

    @staticmethod
    def _get_zero_flows(spec: KernelSpec, block: BlockSpec, connected: set[tuple[str, str]]) -> list[_RelationSpec]:
        """Return the relation `flow == 0` of every unconnected port of an element.

        A port is unconnected when no connection names it.
        """
        if block.node.interaction:
            return []
        ports = getattr(block.node.component_type, 'ports', {})
        relations = []
        for port, connector in ports.items():
            if (block.node.path, port) in connected:
                continue
            flow = getattr(spec.namespace, f'{port}_{connector.flow}')
            description = f'the zero flow of the unconnected port {port} of {spec.owner}'
            relations.append(_RelationSpec(spec, SR(flow), {flow: flow}, description))
        return relations

    @staticmethod
    def _get_specs_of(
        compilation: Compilation,
        orders: _OrdersSpec,
    ) -> list[tuple[KernelSpec, list[BaseLaw | _RelationSpec]]]:
        """Return the laws of every spec with orders and lags read.

        Equations become relations, and zero flows are added.
        """
        connected = OrdersPass._get_connected_ports(compilation)
        return [
            (
                spec,
                [
                    *OrdersPass._copy_with_orders(spec, laws, block, orders),
                    *OrdersPass._get_zero_flows(spec, block, connected),
                ],
            )
            for spec, laws, block in Compilation._get_required(compilation.lowered, 'lowered')
        ]

    @staticmethod
    def _get_program_variables(compilation: Compilation) -> set[str]:
        """Return the dotted names of the algebraic states linear programs determine.

        Linear programs, rather than equations, determine them.
        """
        return {
            reference.entry.name
            for spec, laws in Compilation._get_required(compilation.programs, 'programs')
            for law in laws
            for target in law._get_targets()
            if (reference := spec.references.get(target)) is not None
        }

    @staticmethod
    def _get_algebraic_symbols(blocks: list[BlockSpec], orders: _OrdersSpec) -> list[Expression]:
        """Return the symbols of every algebraic state that equations determine.

        The states are those of the blocks.
        """
        symbols: list[Expression] = []
        for block in blocks:
            if block.node.interaction or block.node.is_agent:
                continue
            declarations = declarations_of(block.node.component_type, VariableKind.STATE)
            symbols += [
                item
                for variable in block.entries
                if variable in declarations and block.entries[variable].name in orders.algebraic
                for item in components_of(block.symbols[variable])
            ]
        return symbols

    @staticmethod
    def _get_unknowns_of(
        specs: list[tuple[KernelSpec, list[BaseLaw | _RelationSpec]]],
        blocks: list[BlockSpec],
        orders: _OrdersSpec,
    ) -> tuple[list[_RelationSpec], dict[Expression, Expression]]:
        """Return the relations with those of one target added, and their unknowns.

        It gives the state of every unknown they solve for.
        """
        relations = OrdersPass._get_summed(
            [item for _, laws in specs for item in laws if isinstance(item, _RelationSpec)],
        )
        unknowns = {item: item for item in OrdersPass._get_algebraic_symbols(blocks, orders)}
        for relation in relations:
            unknowns.update(relation.highest)
        return relations, unknowns

    @staticmethod
    def _get_structure_issues(compilation: Compilation) -> list[IssueSpec]:
        """Return the issue of a system with unmatched algebraic equations.

        Its equations of algebraic states cannot be matched to their unknowns.

        Args:
            compilation:
                The compilation, lowered as far as the pass reads.

        Returns:
            issues:
                The issue, or none for a system that matches or holds no algebraic state.
        """
        blocks = list(Compilation._get_required(compilation.layout, 'layout').blocks.values())
        orders = OrdersPass._get_orders_of(blocks, OrdersPass._get_program_variables(compilation))
        if not orders.algebraic:
            return []
        specs = OrdersPass._get_specs_of(compilation, orders)
        relations, unknowns = OrdersPass._get_unknowns_of(specs, blocks, orders)
        references = {key: value for spec, _ in specs for key, value in spec.references.items()}
        return EquationBlocksLowering._get_structure_issues(relations, unknowns, references, compilation.system.label)

    @staticmethod
    def _get_orders_of(blocks: list[BlockSpec], programmed: set[str]) -> _OrdersSpec:
        """Return the orders above 1 and lags of every state, and the solved states.

        The states are those of the blocks, and the solved states are those equations determine.
        """
        orders, lags, algebraic = {}, {}, set()
        for block in blocks:
            if block.node.interaction or block.node.is_agent:
                continue
            declarations = declarations_of(block.node.component_type, VariableKind.STATE)
            for variable, entry in block.entries.items():
                declaration = declarations.get(variable)
                if declaration is not None and declaration.derivatives:
                    orders[entry.name] = declaration.derivatives + 1
                if declaration is not None and declaration.lags:
                    lags[entry.name] = declaration.lags
                if declaration is not None and declaration.algebraic and entry.name not in programmed:
                    algebraic.add(entry.name)
        return _OrdersSpec(orders, lags, frozenset(algebraic))

    @staticmethod
    def _get_companion_symbol(spec: KernelSpec, item: Expression, suffix: str) -> Expression:
        """Return the symbol of a companion state of the state a symbol reads.

        It is in the same namespace and column.
        """
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

    @staticmethod
    def _substitute(
        value: object,
        spec: KernelSpec,
        orders: _OrdersSpec,
        owner: str,
        highest: dict[Expression, Expression] | None = None,
    ) -> object:
        """Return a law value with derivatives and lags replaced by companion symbols.

        When `highest` is given, derivatives of a state's own order become placeholders that map to the symbol of the
        state's highest derivative, which an equation solves for.
        """
        if isinstance(value, FreeModuleElement):
            return vector(OrdersPass._substitute(item, spec, orders, owner, highest) for item in value)
        if not isinstance(value, Expression):
            return value
        replacements = {}
        for call in BasePass._get_calls_of(value, 'derivative'):
            item, degree = call.operands()
            reference = spec.references.get(item)
            order = orders.orders.get(reference.entry.name, 1) if reference is not None else 1
            if highest is not None and reference is not None and degree.is_integer() and int(degree) == order:
                top = item if order == 1 else OrdersPass._get_companion_symbol(spec, item, f'_d{order - 1}')
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
            replacements[call] = OrdersPass._get_companion_symbol(spec, item, f'_d{int(degree)}')
        for call in BasePass._get_calls_of(value, 'lag'):
            item, steps = call.operands()
            reference = spec.references.get(item)
            kept = orders.lags.get(reference.entry.name, 0) if reference is not None else 0
            if reference is None or not steps.is_integer() or not 0 < int(steps) <= kept:
                error_msg = (
                    f'{owner} uses {call}, but the state keeps {kept} past values. Declare more with `state(lags=...)`.'
                )
                raise ValidationError(error_msg)
            replacements[call] = OrdersPass._get_companion_symbol(spec, item, f'_lag{int(steps)}')
        return value.subs(replacements) if replacements else value

    @staticmethod
    def _retarget(law: RateLaw | NoiseLaw, spec: KernelSpec, orders: _OrdersSpec) -> RateLaw | NoiseLaw:
        """Return a rate or noise law that targets the highest derivative of its state.

        It first checks its order.
        """
        items = components_of(law._get_targets()[0])
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
        tops = [OrdersPass._get_companion_symbol(spec, item, f'_d{order - 1}') for item in items]
        target = vector(tops) if isinstance(law._get_targets()[0], FreeModuleElement) else tops[0]
        return BasePass._get_replaced(law, target=target)

    @staticmethod
    def _get_intrinsic_laws(spec: KernelSpec, block: BlockSpec, orders: _OrdersSpec) -> list[BaseLaw]:
        """Return the chains of higher-order states and the shifts of lagged states.

        The chain of every higher-order state is `x' = x_d1, ...`.
        """
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

    @staticmethod
    def _get_relation(spec: KernelSpec, law: EquationLaw, orders: _OrdersSpec) -> _RelationSpec:
        """Return an equation law as a relation in highest derivative placeholders.

        The placeholders stand for the highest derivatives it contains.
        """
        description = f'the equation {law.lhs_} == {law.rhs_} of {spec.owner}'
        if spec.pairs is not None:
            error_msg = f'Pair interaction {spec.owner} returns {description}. Pair interactions take only rate laws.'
            raise ValidationError(error_msg)
        highest: dict[Expression, Expression] = {}
        lhs = OrdersPass._substitute(SR(law.lhs_), spec, orders, f'In {description}, it', highest)
        rhs = OrdersPass._substitute(SR(law.rhs_), spec, orders, f'In {description}, it', highest)
        reference = spec.references.get(law.target) if isinstance(law.target, Expression) else None
        key = None if reference is None else (reference.entry.name, reference.column)
        target = None if key is None else law.target
        expression = SR(lhs) - SR(rhs)
        highest.update(
            {
                item: item
                for item in expression.variables()
                if item in spec.references and spec.references[item].entry.name in orders.algebraic
            },
        )
        return _RelationSpec(spec, expression, highest, description, key, target)

    @staticmethod
    def _check_relation(relation: _RelationSpec) -> None:
        """Raise an error for a relation with no highest state derivative to solve for."""
        if not relation.highest:
            description = relation.description
            error_msg = (
                f'{description[0].upper()}{description[1:]} contains no highest derivative of a state to solve for.'
            )
            raise ValidationError(error_msg)

    @staticmethod
    def _get_summed(relations: list[_RelationSpec]) -> list[_RelationSpec]:
        """Return the relations with those of one target added into one."""
        groups: dict[tuple[str, int] | int, list[_RelationSpec]] = {}
        for index, relation in enumerate(relations):
            groups.setdefault(index if relation.key is None else relation.key, []).append(relation)
        summed = []
        for group in groups.values():
            first = group[0]
            if len(group) > 1:
                references = {key: value for item in group for key, value in item.spec.references.items()}
                owner = ', '.join(dict.fromkeys(item.spec.owner for item in group))
                spec = KernelSpec(owner, SimpleNamespace(), references, max(item.spec.rows for item in group))
                first = _RelationSpec(
                    spec,
                    sum((item.expression for item in group), SR(0)),
                    {key: value for item in group for key, value in item.highest.items()},
                    ' + '.join(item.description for item in group),
                    first.key,
                    first.target,
                )
            summed.append(first)
        return summed

    @staticmethod
    def _get_ordered_law(spec: KernelSpec, law: BaseLaw, orders: _OrdersSpec) -> BaseLaw:
        """Return a jump, event, renewal, or assignment law with orders lowered.

        Derivatives and lags are replaced and rates retargeted.
        """
        owner = f'A {type(law).__name__} of {spec.owner}'
        if isinstance(law, JumpLaw):
            return JumpLaw(OrdersPass._substitute(law.rate, spec, orders, owner), law.changes)
        if isinstance(law, CrossingLaw | RenewalLaw):
            reset = {
                OrdersPass._substitute(SR(target), spec, orders, owner): (
                    value if BaseLaw._is_function(value) else OrdersPass._substitute(SR(value), spec, orders, owner)
                )
                for target, value in (law.reset or {}).items()
            }
            reset_value = reset if law.reset is not None else None
            if isinstance(law, RenewalLaw):
                return BasePass._get_replaced(
                    law,
                    waiting=(
                        law.waiting
                        if BaseLaw._is_function(law.waiting)
                        else OrdersPass._substitute(SR(law.waiting), spec, orders, owner)
                    ),
                    reset=reset_value,
                )
            return BasePass._get_replaced(
                law,
                guard=(
                    law.guard
                    if BaseLaw._is_function(law.guard)
                    else OrdersPass._substitute(SR(law.guard), spec, orders, owner)
                ),
                reset=reset_value,
            )
        if not isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw):
            return law
        value = OrdersPass._substitute(law.value, spec, orders, owner)
        if isinstance(law, RateLaw | NoiseLaw):
            return OrdersPass._retarget(BasePass._get_replaced(law, value=value), spec, orders)
        return BasePass._get_replaced(law, value=value)

    @staticmethod
    def _copy_with_orders(
        spec: KernelSpec,
        laws: list[BaseLaw],
        block: BlockSpec,
        orders: _OrdersSpec,
    ) -> list[BaseLaw | _RelationSpec]:
        """Return laws with derivatives and lags replaced and rates retargeted.

        Equations become relations, and chains are added.
        """
        if not orders.orders and not orders.lags and not any(isinstance(law, EquationLaw) for law in laws):
            return list(laws)
        result: list[BaseLaw | _RelationSpec] = []
        for law in laws:
            if isinstance(law, KERNEL_FREE_LAWS):
                result.append(law)
            elif isinstance(law, EquationLaw):
                result.append(OrdersPass._get_relation(spec, law, orders))
            else:
                result.append(OrdersPass._get_ordered_law(spec, law, orders))
        if not block.node.interaction:
            result += [
                item
                for law in OrdersPass._get_intrinsic_laws(spec, block, orders)
                for item in LowerComplexPass._split_law(
                    law,
                    spec.owner,
                    spec._get_state_name,
                )
            ]
        return result

    @staticmethod
    def _get_numeric_solution(rows: int, size: int, index: int) -> Callable[..., Array]:
        """Return a function that solves the linear system for one unknown.

        The function takes the matrix and right side entries.
        """

        def solve(t: float, *entries: npt.ArrayLike) -> Array:
            del t
            values = [np.broadcast_to(np.asarray(entry, dtype=float), (rows,)) for entry in entries]
            matrix_ = np.stack(values[: size * size], axis=-1).reshape(rows, size, size)
            right = np.stack(values[size * size :], axis=-1)
            return np.linalg.solve(matrix_, right[..., None])[..., 0][:, index]

        return solve

    @staticmethod
    def _get_checked_relations(specs: list[tuple[KernelSpec, list[BaseLaw | _RelationSpec]]]) -> list[_RelationSpec]:
        """Return the relations of specs with those of one target added.

        Each is checked for a highest derivative.
        """
        relations = OrdersPass._get_summed(
            [item for _, laws in specs for item in laws if isinstance(item, _RelationSpec)],
        )
        for relation in relations:
            OrdersPass._check_relation(relation)
        return relations

    @staticmethod
    def _solve_equations(
        specs: list[tuple[KernelSpec, list[BaseLaw | _RelationSpec]]],
    ) -> list[tuple[KernelSpec, list[BaseLaw]]]:
        """Return specs whose equations are replaced by rate laws on highest derivatives.

        They are the highest derivatives the equations solve for jointly.
        """
        relations = OrdersPass._get_checked_relations(specs)
        kept = [(spec, [law for law in laws if not isinstance(law, _RelationSpec)]) for spec, laws in specs]
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
                            f'derivative {highest[first]}. Write the equation so that highest derivatives appear '
                            'linearly.'
                        )
                        raise ValidationError(error_msg)
        descriptions = ', '.join(relation.description for relation in relations)
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
        merged = KernelSpec(owner, SimpleNamespace(), references, rows)
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
                RateLaw(
                    highest[item],
                    CallSpec(OrdersPass._get_numeric_solution(rows, len(unknowns), index), tuple(entries)),
                )
                for index, item in enumerate(unknowns)
            ]
        return [*kept, (merged, laws)]

    @staticmethod
    def _get_exact_solution(relations: list[_RelationSpec]) -> list[Expression] | None:
        """Return the unique explicit solution of static equations for their targets.

        It is `None` when Sage finds none.
        """
        unknowns = [item.target for item in relations]
        reduced = {
            name for item in relations for call in reductions_in(item.expression) for name in call.call.variables()
        }
        if len(relations) > EQUATION_UNKNOWNS or reduced & set(unknowns):
            return None
        known = {name for item in relations for name in item.expression.variables()}
        try:
            solutions = solve([item.expression == 0 for item in relations], unknowns, solution_dict=True)
        except TypeError:
            return None
        if len(solutions) != 1 or any(item not in solutions[0] for item in unknowns):
            return None
        values = [SR(solutions[0][item]) for item in unknowns]
        explicit = all(set(value.variables()) <= known - set(unknowns) for value in values)
        return values if explicit else None

    @staticmethod
    def _solve_static(
        specs: list[tuple[KernelSpec, list[BaseLaw | _RelationSpec]]],
    ) -> list[tuple[KernelSpec, list[BaseLaw]]]:
        """Return specs whose equations are replaced by updates or residual rates.

        The updates are those of an exact solution, and else the rates are those of their residuals. An equation without
        a state as its target is left out, since the compiler reports it.
        """
        declared = [item for _, laws in specs for item in laws if isinstance(item, _RelationSpec)]
        relations = OrdersPass._get_summed([item for item in declared if item.key is not None])
        kept = [(spec, [law for law in laws if not isinstance(law, _RelationSpec)]) for spec, laws in specs]
        if not relations:
            return kept
        values = OrdersPass._get_exact_solution(relations)
        if values is None:
            return [*kept, *((item.spec, [RateLaw(item.target, item.expression)]) for item in relations)]
        pairs = zip(relations, values, strict=True)
        return [*kept, *((item.spec, [UpdateLaw(item.target, value)]) for item, value in pairs)]
