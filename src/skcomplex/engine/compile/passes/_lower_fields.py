"""Pass that lowers the differential operators of laws to stencils and notes stable steps and linear terms."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression

from ....core.elements import FieldElement
from ....core.laws import BaseLaw, BoundaryLaw, NoiseLaw, RateLaw, UpdateLaw
from ...problems import (
    Reference,
    Source,
)
from ...problems._linear_term import OPERATORS, LinearTerm, Terms
from .._compilation import Compilation, Laws
from .._declarations import Declarations
from .._field_lowering import FieldLowering
from .._kernel_spec import KernelSpec
from .._pipeline import BasePass


class LowerFieldsPass(BasePass):
    """Pass that lowers the differential operators of laws to stencils and notes stable steps and linear terms."""

    reads = ('corrected', 'layout')
    writes = ('corrected', 'written', 'limits', 'volume_limits', 'linear')

    def _run(self, compilation: Compilation) -> None:
        """Lower the differential operators of the corrected laws.

        Args:
            compilation:
                The compilation state.
        """
        declarations = cast('Declarations', compilation.layout)
        facts: dict[str, Any] = {'written': defaultdict(list), 'limits': [], 'volume_limits': [], 'linear': []}
        compilation.corrected = [
            (spec, self._with_fields(spec, laws, declarations, facts))
            for spec, laws in cast('Laws', compilation.corrected)
        ]
        compilation.facts.update(facts)

    @staticmethod
    def _initial_values(spec: KernelSpec, declarations: Declarations) -> dict[Expression, float]:
        """Return the initial value of every symbol of a component, the first member's for groups."""
        states = np.concatenate([np.zeros(0), *declarations.initial_states])
        parameters = np.concatenate([np.zeros(0), *declarations.initial_parameters])
        return {
            item: float((states if reference.is_state else parameters)[reference.entry.start + reference.column])
            for item, reference in spec.references.items()
            if reference.entry.start + reference.column < (states if reference.is_state else parameters).size
        }

    @staticmethod
    def _linear_terms_of(spec: KernelSpec, laws: list[BaseLaw], places: dict[str, FieldElement]) -> list[LinearTerm]:
        """Return the linear terms with constant coefficients of the rates of field states."""
        terms = []
        for law in laws:
            if not isinstance(law, RateLaw) or not isinstance(law.value, Expression):
                continue
            for reference in KernelSpec._target_references(law._get_targets()[0], spec):
                if reference.entry.name in places and reference.source is Source.MEMBER:
                    terms += LowerFieldsPass._linear_terms(law.value, reference, spec.references, places)
        return terms

    @staticmethod
    def _with_fields(
        spec: KernelSpec,
        laws: list[BaseLaw],
        declarations: Declarations,
        facts: dict[str, Any],
    ) -> list[BaseLaw]:
        """Return laws whose differential operators are lowered, noting stable steps and the linear terms of rates."""
        places = LowerFieldsPass._places(declarations)
        facts['linear'] += LowerFieldsPass._linear_terms_of(spec, laws, places)
        if not any(FieldLowering._has_operators(getattr(law, 'value', None)) for law in laws):
            return laws
        lowering = FieldLowering(
            spec.owner,
            spec.references,
            places,
            LowerFieldsPass._initial_values(spec, declarations),
        )
        result: list[BaseLaw] = []
        for law in laws:
            if not isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw) or not FieldLowering._has_operators(
                law.value,
            ):
                result.append(law)
                continue
            if isinstance(law, RateLaw):
                lowering.note_limit(law.value)
            lowered = lowering.lower(law.value)
            if isinstance(law.value, Expression):
                for reference in KernelSpec._target_references(law._get_targets()[0], spec):
                    facts['written'][(reference.entry.name, reference.column)].append((lowered, law.value))
            result.append(BasePass._replaced(law, value=lowered))
        facts['limits'] += lowering.limits
        facts['volume_limits'] += lowering.volume_limits
        return result

    @staticmethod
    def _name_of(operator: object) -> str:
        """Return the name of a Python operator of a Sage expression, or an empty string."""
        return getattr(operator, '__name__', '')

    @staticmethod
    def _sum_terms(operands: list[Any], states: Mapping[Expression, Any]) -> Terms | None:
        """Return the linear terms of a sum, or `None` when a summand is not linear."""
        terms: Terms = []
        for item in operands:
            part = LowerFieldsPass._linear(item, states)
            if part is None:
                return None
            terms += part
        return terms

    @staticmethod
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
        part = LowerFieldsPass._linear(variable[0], states)
        return None if part is None else [(constant * coefficient, chain, state) for coefficient, chain, state in part]

    @staticmethod
    def _operator_terms(name: str, operands: list[Any], states: Mapping[Expression, Any]) -> Terms | None:
        """Return the linear terms of an operator applied to a linear expression, or `None`."""
        inner = LowerFieldsPass._linear(operands[0], states)
        if inner is None:
            return None
        axis = int(operands[1]) if name == 'partial' else 0
        return [(coefficient, ((name, axis), *chain), state) for coefficient, chain, state in inner]

    @staticmethod
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
            return LowerFieldsPass._operator_terms(name, expression.operands(), states)
        combine = {'add_vararg': LowerFieldsPass._sum_terms, 'mul_vararg': LowerFieldsPass._product_terms}.get(
            LowerFieldsPass._name_of(operator),
        )
        return None if combine is None else combine(expression.operands(), states)

    @staticmethod
    def _linear_terms(
        value: object,
        target: Reference,
        references: Mapping[Expression, Reference],
        fields: Mapping[str, FieldElement],
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
        parts = expanded.operands() if LowerFieldsPass._name_of(expanded.operator()) == 'add_vararg' else [expanded]
        terms = []
        for part in parts:
            found = LowerFieldsPass._linear(part, states)
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

    @staticmethod
    def _places(declarations: Declarations) -> dict[str, FieldElement]:
        """Return the fields by the name of each of their entries."""
        return {
            entry.name: block.node.component
            for block in declarations.blocks.values()
            if isinstance(block.node.component, FieldElement)
            for entry in block.entries.values()
        }
