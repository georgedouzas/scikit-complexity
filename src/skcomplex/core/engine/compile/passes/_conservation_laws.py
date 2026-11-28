"""Read the rates of finite volume fields as conservation laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import SR, Expression
from sage.symbolic.operators import add_vararg, mul_vararg

from ....components.laws import BaseLaw, RateLaw
from ....components.laws._ordered_rate import _OrderedRateLaw
from ....errors import ValidationError
from ...problems._field_element import _FieldElement
from ...problems._kernel_spec import KernelSpec
from .._base import BasePass
from .._compilation import Compilation, _Declarations
from .._flux_law import _FluxLaw


class ConservationLawsPass(BasePass):
    """A pass that reads the rates of finite volume fields as conservation laws.

    A rate of a field on finite volumes that holds `-divergence(flux)`, such as the transport of Burgers' equation, says
    that the field moves only by its flux. The pass splits such a rate into the flux, which the finite volume solver
    moves between cells, and a rate of the other terms, which are sources.

    Read more in the user guide, under Extend the library.

    See Also:
        FieldFunctionsPass: Adds the finite volumes to the equations.
        LowerFieldsPass: Rewrites the other operators of fields.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.physics.fluids import BurgersSystem
        >>> compilation = PassCompiler().prepare(BurgersSystem()).compilation_
        >>> [law.flux for _, laws, _ in compilation.collected for law in laws]
        [[1/2*u_line^2]]
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('collected', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('collected',)

    def _run(self, inputs: object = None) -> None:
        """Replace the rate laws of finite-volume fields that hold a divergence."""
        compilation = self.compilation_
        declarations = Compilation._get_required(compilation.layout, 'layout')
        compilation.collected = [
            (spec, [item for law in laws for item in self._split(spec, law, declarations)], block)
            for spec, laws, block in Compilation._get_required(compilation.collected, 'collected')
        ]

    @staticmethod
    def _get_field(spec: KernelSpec, law: RateLaw, declarations: _Declarations) -> _FieldElement | None:
        """Return the finite-volume field whose scalar state a rate law targets, or `None`."""
        scalar = isinstance(law.target, Expression) and isinstance(law.value, Expression)
        if law._get_order() not in (None, 1) or not scalar:
            return None
        references = [item for item in spec._get_target_references(law.target) if item is not None]
        if len(references) != 1:
            return None
        owner = references[0].entry.name.rpartition('.')[0]
        block = declarations.blocks.get(owner)
        place = None if block is None else block.node.component
        if not isinstance(place, _FieldElement) or place.discretization != 'finite-volume':
            return None
        return place

    @staticmethod
    def _get_partial(term: Expression) -> tuple[Expression, Expression, int] | None:
        """Return the factor, operand, and axis of a term `c * partial(f, k)`, or `None`."""
        factors = term.operands() if term.operator() is mul_vararg else [term]
        calls = [item for item in factors if BasePass._get_calls_of(item, 'partial') and item.operator() is not None]
        if len(calls) != 1 or str(calls[0].operator()) != 'partial':
            return None
        coefficient = SR(1)
        for item in factors:
            if item is not calls[0]:
                coefficient *= item
        if coefficient.variables():
            return None
        operand, axis = calls[0].operands()
        return coefficient, operand, int(axis)

    @staticmethod
    def _split(spec: KernelSpec, law: BaseLaw, declarations: _Declarations) -> list[BaseLaw]:
        """Return a rate law as a flux law and a rate law of its sources.

        Any other law, and a rate law that is no conservation law, stays as it is.

        Args:
            spec:
                The kernel spec that holds the law.

            law:
                The law to split.

            declarations:
                The declarations of the values of the system.

        Returns:
            laws:
                The flux law and the rate law of the sources, or a list of the law itself.

        Raises:
            ValidationError:
                The rate applies `divergence` to a flux in a way other than a sum with constant factors.
        """
        if not isinstance(law, RateLaw):
            return [law]
        place = ConservationLawsPass._get_field(spec, law, declarations)
        if place is None or place.space is None or not BasePass._get_calls_of(law.value, 'partial'):
            return [law]
        value = SR(law.value)
        terms = value.operands() if value.operator() is add_vararg else [value]
        n_dim = int(place.space.prepare().n_dim_)
        flux = [SR(0)] * n_dim
        rest = SR(0)
        for term in terms:
            found = ConservationLawsPass._get_partial(term)
            if found is None:
                if BasePass._get_calls_of(term, 'partial'):
                    error_msg = (
                        f'A rate law of {spec.owner} on the finite-volume field {place.label} reads {term}. Write the '
                        'transport as `-divergence(flux)` plus sources.'
                    )
                    raise ValidationError(error_msg)
                rest += term
                continue
            coefficient, operand, axis = found
            flux[axis] -= coefficient * operand
        scheme = place._get_scheme()
        laws: list[BaseLaw] = [
            _FluxLaw(
                law.target,
                flux,
                limiter=scheme.limiter,
                riemann=scheme.riemann,
                speed=scheme.speed,
                label=law.label,
            ),
        ]
        if not rest.is_zero():
            laws.append(_OrderedRateLaw(law.target, rest, law._get_order(), label=law.label))
        return laws
