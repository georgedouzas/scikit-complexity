"""Pass that checks the Hurst index and amplitude of fractional noise laws, with Hurst index 1/2 as noise laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....components.laws import BaseLaw, FractionalNoiseLaw, NoiseLaw
from ....errors import ValidationError
from ..._utils import components_of, symbols_of
from ...problems._declarations_spec import DeclarationsSpec
from ...problems._kernel_spec import KernelSpec
from .._base import BasePass
from .._compilation import Compilation

BROWNIAN_HURST = 0.5


class CheckFractionalNoisePass(BasePass):
    """Pass that checks the Hurst index and amplitude of fractional noise laws, with Hurst index 1/2 as noise laws."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('lowered', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('lowered',)

    def _run(self, inputs: object = None) -> None:
        """Check the fractional noise laws."""
        compilation = self.compilation_
        declarations = Compilation._required(compilation.layout, 'layout')
        compilation.lowered = [
            (spec, self._checked_fractional(spec, laws, declarations), block)
            for spec, laws, block in Compilation._required(compilation.lowered, 'lowered')
        ]

    @staticmethod
    def _checked_hurst(
        spec: KernelSpec,
        law: FractionalNoiseLaw,
        declarations: DeclarationsSpec,
        description: str,
    ) -> None:
        """Raise an error when the Hurst index of a fractional noise law is not a number or a parameter in `(0, 1)`."""
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
            values = np.asarray(reference.column_of(np.zeros(0), parameters), dtype=float)
        if not np.all((values > 0) & (values < 1)):
            error_msg = (
                f'The {description} of {spec.owner} has the Hurst index {hurst} = {values[0]}, outside the open '
                'interval (0, 1). Give a Hurst index between 0 and 1.'
            )
            raise ValidationError(error_msg)

    @staticmethod
    def _checked_fractional(spec: KernelSpec, laws: list[BaseLaw], declarations: DeclarationsSpec) -> list[BaseLaw]:
        """Return laws with every fractional noise law of Hurst index 1/2 as a noise law, after checking the others."""
        result: list[BaseLaw] = []
        for law in laws:
            if not isinstance(law, FractionalNoiseLaw):
                result.append(law)
                continue
            description = CheckFractionalNoisePass._fractional_description(spec, law)
            CheckFractionalNoisePass._checked_hurst(spec, law, declarations, description)
            if SR(law.hurst).is_numeric() and float(SR(law.hurst)) == BROWNIAN_HURST:
                result.append(NoiseLaw(law._get_targets()[0], law.value, law.order))
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
                    f'{spec._state_name(read[0])}. Only state-independent amplitudes are supported for a '
                    'Hurst index other than 1/2.'
                )
                raise ValidationError(error_msg)
            result.append(law)
        return result

    @staticmethod
    def _fractional_description(spec: KernelSpec, law: FractionalNoiseLaw) -> str:
        """Return the description of a fractional noise law by the states it drives."""
        states = dict.fromkeys(spec._state_name(item) for item in components_of(law._get_targets()[0]))
        return f'fractional noise law of {", ".join(states)}'
