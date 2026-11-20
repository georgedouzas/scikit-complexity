"""Pass that lowers the differential operators of laws to stencils."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from collections.abc import Mapping

from sage.all__sagemath_symbolics import SR, Expression

from ....components.laws import BaseLaw, RateLaw
from ....components.laws._neighbor_reduction_spec import _NeighborReductionSpec
from ....components.laws.utils import laplacian
from ....components.laws.utils._phase import COORDINATE_PREFIX
from ....components.laws.utils._reductions import neighborhoods_in
from ....components.spaces import BaseNeighborhood, CellComplexSpace, LatticeSpace
from ...enums import LinearOperatorKind, ReadKind
from ...problems._diffusive_limit_spec import DiffusiveLimitSpec
from ...problems._field_element import _FieldElement
from ...problems._kernel_spec import KernelSpec
from ...problems._linear_term_spec import LinearTermSpec
from ...problems._reference_spec import ReferenceSpec
from ...types import Terms
from .._base import BasePass
from .._compilation import Compilation
from .._field_lowering import FieldLowering


class LowerFieldsPass(BasePass):
    """A pass that lowers the differential operators of laws to stencils.

    It notes stable steps and linear terms.
    """

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
        corrected = compilation.corrected = list(Compilation._get_required(compilation.corrected, 'corrected'))
        for index, (spec, laws) in enumerate(corrected):
            lowered = self._copy_with_fields(
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
    def _get_linear_terms_of(
        spec: KernelSpec,
        laws: list[BaseLaw],
        places: dict[str, _FieldElement],
    ) -> list[LinearTermSpec]:
        """Return the linear terms with constant coefficients of the rates of field states."""
        terms = []
        for law in laws:
            if not isinstance(law, RateLaw) or not isinstance(law.value, Expression):
                continue
            for reference in spec._get_target_references(law._get_targets()[0]):
                if reference.entry.name in places and reference.source is ReadKind.MEMBER:
                    terms += LowerFieldsPass._get_linear_terms(law.value, reference, spec.references, places)
        return terms

    @staticmethod
    def _copy_with_fields(
        spec: KernelSpec,
        laws: list[BaseLaw],
        compilation: Compilation,
        *,
        written: defaultdict[tuple[str, int], list[tuple[Expression, Expression]]],
        limits: list[float],
        volume_limits: list[DiffusiveLimitSpec],
        linear: list[LinearTermSpec],
    ) -> list[BaseLaw]:
        """Return laws whose differential operators are lowered.

        The stable steps and the linear terms of rates are noted. The spec is replaced by a copy with the references of
        the neighbour reads and derived values in the compilation.
        """
        places = FieldLowering._get_field_entries(Compilation._get_required(compilation.layout, 'layout'))
        laws = [LowerFieldsPass._get_with_laplacians(spec, law, places) for law in laws]
        linear += LowerFieldsPass._get_linear_terms_of(spec, laws, places)
        if not any(FieldLowering._get_law_has_operators(law) for law in laws):
            return laws
        lowering = FieldLowering(spec).prepare(compilation)
        result = lowering.run(laws).lowered_
        lowering._apply_references()
        for key, items in lowering.written_.items():
            written[key] += items
        limits += lowering.limits_
        volume_limits += lowering.volume_limits_
        return result

    @staticmethod
    def _get_with_laplacians(spec: KernelSpec, law: BaseLaw, places: Mapping[str, _FieldElement]) -> BaseLaw:
        """Return a rate whose diffusive neighbour sums are Laplacians.

        The sums are those on a spectral lattice or a mesh.

        A sum over the nearest neighbours of a constant times the difference `u_j - u_i` of a state is the Laplacian of
        the state times the squared spacing of a spectral lattice, read in Fourier space, and the finite element
        Laplacian on a mesh, as `D * neighbors(s.other.u - s.u)` is `D * laplacian(u)` on the edges of a mesh.
        """
        if not isinstance(law, RateLaw) or not isinstance(law.value, Expression):
            return law
        replacements = {}
        for reduction in neighborhoods_in(law.value):
            found = LowerFieldsPass._get_difference(spec, reduction, places)
            if found is not None:
                coefficient, member, scale = found
                replacements[reduction.call] = coefficient * scale * laplacian(member)
        if not replacements:
            return law
        return BasePass._get_replaced(law, value=SR(law.value).subs(replacements))

    @staticmethod
    def _get_difference(
        spec: KernelSpec,
        reduction: _NeighborReductionSpec,
        places: Mapping[str, _FieldElement],
    ) -> tuple[Expression, Expression, float] | None:
        """Return the coefficient, the state, and the scale of a diffusive neighbour sum.

        The sum is one on a spectral field or a mesh.

        It is `None` for another reduction, such as one over farther neighbours or one of a function of both ends.
        """
        argument = SR(reduction.argument)
        others = [item for item in argument.variables() if LowerFieldsPass._get_source(spec, item) is ReadKind.OTHER]
        if reduction.kind != 'sum' or reduction.weight is not None or len(others) != 1:
            return None
        other = spec.references[others[0]]
        member = next(
            (
                item
                for item, reference in spec.references.items()
                if reference.source is ReadKind.MEMBER
                and reference.is_state
                and (reference.entry.name, reference.column) == (other.entry.name, other.column)
            ),
            None,
        )
        place = places.get(other.entry.name)
        if member is None or place is None or not other.is_state:
            return None
        coefficient = SR(argument.diff(others[0]))
        if any(LowerFieldsPass._get_source(spec, item) is not None for item in coefficient.variables()):
            return None
        if not SR(argument - coefficient * (others[0] - member)).expand().is_zero():
            return None
        scale = LowerFieldsPass._get_scale(place, reduction.over)
        return None if scale is None else (coefficient, member, scale)

    @staticmethod
    def _get_source(spec: KernelSpec, item: Expression) -> ReadKind | None:
        """Return how a state symbol is read, or `None` for any other symbol."""
        reference = spec.references.get(item)
        return reference.source if reference is not None and reference.is_state else None

    @staticmethod
    def _get_scale(place: _FieldElement, over: object) -> float | None:
        """Return the factor from the Laplacian of a field to its neighbour differences.

        The differences are summed over the neighbours.

        It is the squared spacing of a spectral lattice with one spacing summed over its nearest sites, 1 on a mesh
        summed over its edges, and `None` for other fields and neighbourhoods.
        """
        space = place.space
        if place.discretization == 'finite-element' and isinstance(space, CellComplexSpace) and space._is_mesh():
            return 1.0 if over is None else None
        if place.discretization != 'spectral' or not isinstance(space, LatticeSpace):
            return None
        space = space.prepare()
        nearest = {
            tuple(sign * int(axis == other) for other in range(space.n_dim_))
            for axis in range(space.n_dim_)
            for sign in (-1, 1)
        }
        neighborhood = space._get_default_neighborhood() if over is None else over
        offsets = neighborhood._get_offsets(space.n_dim_) if isinstance(neighborhood, BaseNeighborhood) else None
        spacings = set(space.spacings_)
        if offsets is None or set(offsets) != nearest or len(offsets) != len(nearest) or len(spacings) != 1:
            return None
        return spacings.pop() ** 2

    @staticmethod
    def _get_name_of(operator: object) -> str:
        """Return the name of a Python operator of a Sage expression, or an empty string."""
        return getattr(operator, '__name__', '')

    @staticmethod
    def _sum_terms(operands: list[Expression], states: Mapping[Expression, ReferenceSpec]) -> Terms | None:
        """Return the linear terms of a sum, or `None` when a summand is not linear."""
        terms: Terms = []
        for item in operands:
            part = LowerFieldsPass._get_linear(item, states)
            if part is None:
                return None
            terms += part
        return terms

    @staticmethod
    def _get_product_terms(operands: list[Expression], states: Mapping[Expression, ReferenceSpec]) -> Terms | None:
        """Return the linear terms of a constant times a linear expression, or `None`."""
        constant, variable = SR(1), []
        for item in operands:
            if any(symbol in states for symbol in item.variables()):
                variable.append(item)
            else:
                constant *= item
        if len(variable) != 1:
            return None
        part = LowerFieldsPass._get_linear(variable[0], states)
        return None if part is None else [(constant * coefficient, chain, state) for coefficient, chain, state in part]

    @staticmethod
    def _get_operator_terms(
        name: str,
        operands: list[Expression],
        states: Mapping[Expression, ReferenceSpec],
    ) -> Terms | None:
        """Return the linear terms of an operator applied to a linear expression.

        It is `None` otherwise.
        """
        inner = LowerFieldsPass._get_linear(operands[0], states)
        if inner is None or (name == 'inverse_helmholtz' and not SR(operands[1]).is_zero()):
            return None
        axis = int(operands[1]) if name == 'partial' else 0
        return [(coefficient, ((name, axis), *chain), state) for coefficient, chain, state in inner]

    @staticmethod
    def _get_linear(expression: object, states: Mapping[Expression, ReferenceSpec]) -> Terms | None:
        """Return the linear terms of an expression in field states.

        It is `None` when the expression is not linear.
        """
        value = SR(expression)
        if not any(item in states for item in value.variables()):
            return []
        if value in states:
            return [(SR(1), (), value)]
        operator = value.operator()
        name = getattr(operator, 'name', lambda: '')()
        if name in LinearOperatorKind:
            return LowerFieldsPass._get_operator_terms(name, value.operands(), states)
        combine = {'add_vararg': LowerFieldsPass._sum_terms, 'mul_vararg': LowerFieldsPass._get_product_terms}.get(
            LowerFieldsPass._get_name_of(operator),
        )
        return None if combine is None else combine(value.operands(), states)

    @staticmethod
    def _get_linear_terms(
        value: object,
        target: ReferenceSpec,
        references: Mapping[Expression, ReferenceSpec],
        fields: Mapping[str, _FieldElement],
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
        parts = expanded.operands() if LowerFieldsPass._get_name_of(expanded.operator()) == 'add_vararg' else [expanded]
        parts = [
            part for part in parts if not any(str(item).startswith(COORDINATE_PREFIX) for item in part.variables())
        ]
        terms = []
        for part in parts:
            found = LowerFieldsPass._get_linear(part, states)
            for coefficient, chain, state in found or []:
                if any(item not in references or references[item].is_state for item in coefficient.variables()):
                    continue
                source = states[state]
                place = fields[source.entry.name]
                if place.discretization != 'spectral' and any(name == 'inverse_helmholtz' for name, _ in chain):
                    continue
                terms.append(
                    LinearTermSpec(
                        target,
                        source,
                        place,
                        chain,
                        coefficient,
                        {item: references[item] for item in coefficient.variables()},
                    ),
                )
        return terms
