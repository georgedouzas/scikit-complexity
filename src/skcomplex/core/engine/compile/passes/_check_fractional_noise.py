"""Check the Hurst index and amplitude of fractional noise laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....components.laws import BaseLaw, NoiseLaw
from ....components.laws._ordered_noise import _OrderedNoiseLaw
from ....errors import ValidationError
from ..._utils import components_of, symbols_of
from ...problems._kernel_spec import KernelSpec
from .._base import BasePass
from .._compilation import Compilation, _Declarations

BROWNIAN_HURST = 0.5


class CheckFractionalNoisePass(BasePass):
    """A pass that checks the Hurst index and amplitude of fractional noise laws.

    A noise law with a Hurst index `H` drives a state with fractional Brownian motion, such as the walk of a molecule in
    a crowded cell. The pass checks that `H` lies in `(0, 1)`, and a law of `H = 1/2`, plain Brownian motion, becomes a
    noise law without an index.

    Read more in the user guide, under Extend the library.

    See Also:
        FractionalProblemPass: Adds the fractional noise to the problem.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.physics.soft_matter import FractionalBrownianMotionSystem
        >>> compilation = PassCompiler().prepare(FractionalBrownianMotionSystem()).compilation_
        >>> [law.hurst for _, laws, _ in compilation.lowered for law in laws]
        [H_noise]
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('lowered', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('lowered',)

    def _run(self, inputs: object = None) -> None:
        """Check the noise laws with a Hurst index."""
        compilation = self.compilation_
        declarations = Compilation._get_required(compilation.layout, 'layout')
        compilation.lowered = [
            (spec, self._get_checked_fractional(spec, laws, declarations), block)
            for spec, laws, block in Compilation._get_required(compilation.lowered, 'lowered')
        ]

    @staticmethod
    def _get_checked_hurst(
        spec: KernelSpec,
        law: NoiseLaw,
        declarations: _Declarations,
        description: str,
    ) -> None:
        """Raise an error for a bad Hurst index of a noise law.

        The index must be a number or a parameter in `(0, 1)`.
        """
        hurst = SR(law.hurst)
        if hurst.is_numeric():
            values = np.asarray([float(hurst)])
        else:
            reference = spec.references.get(hurst)
            if reference is None or reference.is_state:
                error_msg = (
                    f'The {description} of {spec.owner} has the Hurst index {hurst}. Give a number or a parameter.'
                )
                raise ValidationError(error_msg)
            parameters = np.concatenate([np.zeros(0), *declarations.initial_parameters])
            values = np.asarray(reference.get_column_of(np.zeros(0), parameters), dtype=float)
        if not np.all((values > 0) & (values < 1)):
            error_msg = (
                f'The {description} of {spec.owner} has the Hurst index {hurst} = {values[0]}, outside the open '
                'interval (0, 1). Give a Hurst index between 0 and 1.'
            )
            raise ValidationError(error_msg)

    @staticmethod
    def _get_checked_fractional(spec: KernelSpec, laws: list[BaseLaw], declarations: _Declarations) -> list[BaseLaw]:
        """Return laws with every noise law of Hurst index 1/2 as a noise law without one.

        The others are checked first.
        """
        result: list[BaseLaw] = []
        for law in laws:
            if not isinstance(law, NoiseLaw) or law.hurst is None:
                result.append(law)
                continue
            description = CheckFractionalNoisePass._get_fractional_description(spec, law)
            CheckFractionalNoisePass._get_checked_hurst(spec, law, declarations, description)
            if SR(law.hurst).is_numeric() and float(SR(law.hurst)) == BROWNIAN_HURST:
                result.append(
                    _OrderedNoiseLaw(law._get_targets()[0], law.value, law._get_order(), convention=law.convention),
                )
                continue
            values = [item for item in components_of(law.value) if isinstance(item, Expression)]
            if not isinstance(law.value, Expression | FreeModuleElement) and not SR(law.value).is_numeric():
                error_msg = (
                    f'The {description} of {spec.owner} has an amplitude that is not symbolic. Give an expression.'
                )
                raise ValidationError(error_msg)
            read = [
                item
                for item in sorted(symbols_of(values), key=str)
                if (reference := spec.references.get(item)) is not None and reference.is_state
            ]
            if read:
                error_msg = (
                    f'The {description} of {spec.owner} has an amplitude that depends on the state '
                    f'{spec._get_state_name(read[0])}. Only state-independent amplitudes are supported for a '
                    'Hurst index other than 1/2.'
                )
                raise ValidationError(error_msg)
            result.append(law)
        return result

    @staticmethod
    def _get_fractional_description(spec: KernelSpec, law: NoiseLaw) -> str:
        """Describe a noise law with a Hurst index by the states it drives."""
        states = dict.fromkeys(spec._get_state_name(item) for item in components_of(law._get_targets()[0]))
        return f'fractional noise law of {", ".join(states)}'
