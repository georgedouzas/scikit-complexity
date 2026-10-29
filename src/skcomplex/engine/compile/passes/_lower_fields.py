"""Pass that lowers the differential operators of laws to stencils and notes stable steps and linear terms."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from collections.abc import Mapping

from sage.all__sagemath_symbolics import SR, Expression

from ....core.elements import FieldElement
from ....core.laws import BaseLaw, RateLaw
from ...problems import DiffusiveLimitSpec, KernelSpec, LinearTermSpec, ReferenceSpec
from ...problems._linear_term_spec import OPERATORS
from ...types import ReadKind, Terms
from .._base import BasePass
from .._compilation import Compilation
from .._field_lowering import FieldLowering


class LowerFieldsPass(BasePass):
    """Pass that lowers the differential operators of laws to stencils and notes stable steps and linear terms."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('corrected', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('corrected', 'written', 'limits', 'volume_limits', 'linear')

    def _run(self, inputs: object = None) -> None:
        """Lower the differential operators of the corrected laws.

        Lowering the laws of a spec replaces the spec in the corrected laws, so their entry is read again.
        """
        compilation = self.compilation_
        written: defaultdict[tuple[str, int], list[tuple[Expression, Expression]]] = defaultdict(list)
        limits: list[float] = []
        volume_limits: list[DiffusiveLimitSpec] = []
        linear: list[LinearTermSpec] = []
        corrected = compilation.corrected = list(Compilation._required(compilation.corrected, 'corrected'))
        for index, (spec, laws) in enumerate(corrected):
            lowered = self._with_fields(
                spec,
                laws,
                compilation,
                written=written,
                limits=limits,
                volume_limits=volume_limits,
                linear=linear,
            )
            corrected[index] = (corrected[index][0], lowered)
        compilation.written = written
        compilation.limits = limits
        compilation.volume_limits = volume_limits
        compilation.linear = linear

    @staticmethod
    def _linear_terms_of(
        spec: KernelSpec,
        laws: list[BaseLaw],
        places: dict[str, FieldElement],
    ) -> list[LinearTermSpec]:
        """Return the linear terms with constant coefficients of the rates of field states."""
        terms = []
        for law in laws:
            if not isinstance(law, RateLaw) or not isinstance(law.value, Expression):
                continue
            for reference in spec._target_references(law._get_targets()[0]):
                if reference.entry.name in places and reference.source is ReadKind.MEMBER:
                    terms += LowerFieldsPass._linear_terms(law.value, reference, spec.references, places)
        return terms

    @staticmethod
    def _with_fields(
        spec: KernelSpec,
        laws: list[BaseLaw],
        compilation: Compilation,
        *,
        written: defaultdict[tuple[str, int], list[tuple[Expression, Expression]]],
        limits: list[float],
        volume_limits: list[DiffusiveLimitSpec],
        linear: list[LinearTermSpec],
    ) -> list[BaseLaw]:
        """Return laws whose differential operators are lowered, noting stable steps and the linear terms of rates.

        The spec is replaced by a copy with the references of the neighbour reads and derived values in the
        compilation.
        """
        places = FieldLowering._field_entries(Compilation._required(compilation.layout, 'layout'))
        linear += LowerFieldsPass._linear_terms_of(spec, laws, places)
        if not any(FieldLowering._has_operators(getattr(law, 'value', None)) for law in laws):
            return laws
        lowering = FieldLowering(spec).check().prepare(compilation)
        result = lowering.run(laws).lowered_
        lowering._apply_references()
        for key, items in lowering.written_.items():
            written[key] += items
        limits += lowering.limits_
        volume_limits += lowering.volume_limits_
        return result

    @staticmethod
    def _name_of(operator: object) -> str:
        """Return the name of a Python operator of a Sage expression, or an empty string."""
        return getattr(operator, '__name__', '')

    @staticmethod
    def _sum_terms(operands: list[Expression], states: Mapping[Expression, ReferenceSpec]) -> Terms | None:
        """Return the linear terms of a sum, or `None` when a summand is not linear."""
        terms: Terms = []
        for item in operands:
            part = LowerFieldsPass._linear(item, states)
            if part is None:
                return None
            terms += part
        return terms

    @staticmethod
    def _product_terms(operands: list[Expression], states: Mapping[Expression, ReferenceSpec]) -> Terms | None:
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
    def _operator_terms(
        name: str,
        operands: list[Expression],
        states: Mapping[Expression, ReferenceSpec],
    ) -> Terms | None:
        """Return the linear terms of an operator applied to a linear expression, or `None`."""
        inner = LowerFieldsPass._linear(operands[0], states)
        if inner is None:
            return None
        axis = int(operands[1]) if name == 'partial' else 0
        return [(coefficient, ((name, axis), *chain), state) for coefficient, chain, state in inner]

    @staticmethod
    def _linear(expression: object, states: Mapping[Expression, ReferenceSpec]) -> Terms | None:
        """Return the linear terms of an expression in field states, or `None` when it is not linear."""
        value = SR(expression)
        if not any(item in states for item in value.variables()):
            return []
        if value in states:
            return [(SR(1), (), value)]
        operator = value.operator()
        name = getattr(operator, 'name', lambda: '')()
        if name in OPERATORS:
            return LowerFieldsPass._operator_terms(name, value.operands(), states)
        combine = {'add_vararg': LowerFieldsPass._sum_terms, 'mul_vararg': LowerFieldsPass._product_terms}.get(
            LowerFieldsPass._name_of(operator),
        )
        return None if combine is None else combine(value.operands(), states)

    @staticmethod
    def _linear_terms(
        value: object,
        target: ReferenceSpec,
        references: Mapping[Expression, ReferenceSpec],
        fields: Mapping[str, FieldElement],
    ) -> list[LinearTermSpec]:
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
            if reference.is_state and reference.source is ReadKind.MEMBER and reference.entry.name in fields
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
                    LinearTermSpec(
                        target,
                        source,
                        fields[source.entry.name],
                        chain,
                        coefficient,
                        {item: references[item] for item in coefficient.variables()},
                    ),
                )
        return terms
