"""Add to the drift the correction of every noise law of a convention other than Itô."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....core.errors import ValidationError
from ....core.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.laws._noise import NOISE_CONVENTIONS
from ....core.specs import WienerProcessSpec
from ...problems import components_of
from .._compilation import Compilation, Laws
from .._declarations import Declarations
from .._noise import ProcessKey
from .._pipeline import BasePass
from .._specs import KernelSpec


@dataclass(frozen=True)
class _NoiseTerm:
    """Amplitude of a noise law on one state component, with the process it draws."""

    spec: KernelSpec
    law: NoiseLaw
    target: Expression
    state: tuple[str, int]
    amplitude: object
    process: object


class NoisePass(BasePass):
    """Pass that adds the drift correction of every noise law of a convention after it, in the Itô form."""

    reads = ('delayed', 'layout')
    writes = ('corrected',)

    def _run(self, compilation: Compilation) -> None:
        """Add the drift corrections of the noise laws.

        Args:
            compilation:
                The compilation state.
        """
        compilation.corrected = self._with_noise_corrections(
            cast('Laws', compilation.delayed),
            cast('Declarations', compilation.layout),
        )

    @staticmethod
    def _reads_state(term: _NoiseTerm, state: tuple[str, int]) -> bool:
        """Check whether the amplitude of a noise term depends on a state component."""
        amplitude = SR(term.amplitude)
        return any(
            (reference.entry.name, reference.column) == state and not amplitude.diff(item).is_zero()
            for item in amplitude.variables()
            if (reference := term.spec.references.get(item)) is not None and reference.is_state
        )

    @staticmethod
    def _noise_correction(
        term: _NoiseTerm,
        terms: list[_NoiseTerm],
        wieners: Mapping[str, WienerProcessSpec],
    ) -> BaseLaw | None:
        """Return the drift correction `alpha sum C_jl G_kl dG_ij/dx_k` of a noise term of a convention, or `None`."""
        alpha = NOISE_CONVENTIONS[term.law.convention]
        if not alpha or not isinstance(term.amplitude, Expression | int | float):
            return None
        amplitude = SR(term.amplitude)
        parts = []
        for other in terms:
            covariance = NoisePass._process_covariance(term.process, other.process, wieners)
            if not covariance:
                continue
            if other.spec is not term.spec:
                if NoisePass._reads_state(term, other.state):
                    error_msg = (
                        f'The {term.law.convention} noise law of {term.spec.owner} depends on {other.state[0]}, which '
                        f'a noise law of {other.spec.owner} drives through a shared process, so its drift correction '
                        'is out of scope. Use the Itô convention, or write both laws in one component.'
                    )
                    raise ValidationError(error_msg)
                continue
            derivative = amplitude.diff(other.target)
            if derivative.is_zero():
                continue
            if not isinstance(other.amplitude, Expression | int | float):
                error_msg = (
                    f'The {term.law.convention} noise law of {term.spec.owner} shares a process with a noise law whose '
                    'amplitude is not symbolic, so its drift correction is unknown. Write the amplitude as an '
                    'expression.'
                )
                raise ValidationError(error_msg)
            parts.append(alpha * covariance * SR(other.amplitude) * derivative)
        if not parts:
            return None
        correction = sum(parts[1:], parts[0])
        return None if correction.is_zero() else RateLaw(term.target, correction)

    @staticmethod
    def _with_noise_corrections(
        specs: list[tuple[KernelSpec, list[BaseLaw]]],
        declarations: Declarations,
    ) -> list[tuple[KernelSpec, list[BaseLaw]]]:
        """Return laws with the drift correction of every noise law of a convention after it, in the Itô form."""
        terms = NoisePass._noise_terms(specs, declarations)
        by_law: dict[int, list[_NoiseTerm]] = defaultdict(list)
        for term in terms:
            by_law[id(term.law)].append(term)
        result = []
        for spec, laws in specs:
            corrected: list[BaseLaw] = []
            for law in laws:
                corrected.append(law)
                for term in by_law.get(id(law), []):
                    correction = NoisePass._noise_correction(term, terms, declarations.wieners)
                    if correction is not None:
                        corrected.append(correction)
            result.append((spec, corrected))
        return result

    @staticmethod
    def _noise_terms(specs: list[tuple[KernelSpec, list[BaseLaw]]], declarations: Declarations) -> list[_NoiseTerm]:
        """Return every component of every noise law, an independent process standing for itself alone."""
        terms = []
        for spec, laws in specs:
            for law in laws:
                if not isinstance(law, NoiseLaw):
                    continue
                targets = components_of(law._get_targets()[0])
                values = (
                    components_of(law.value)
                    if isinstance(law.value, FreeModuleElement | list | tuple)
                    else [law.value] * len(targets)
                )
                keys = NoisePass._process_keys(spec, law, declarations)
                for index, (target, value, key) in enumerate(zip(targets, values, keys, strict=True)):
                    reference = spec.references[target]
                    process = key if key is not None else ('independent', id(law), index)
                    terms.append(
                        _NoiseTerm(spec, law, target, (reference.entry.name, reference.column), value, process),
                    )
        return terms

    @staticmethod
    def _process_key(spec: KernelSpec, item: object, declarations: Declarations) -> ProcessKey:
        """Return the process a label or a handle of a noise law names, as a local label or a Wiener process."""
        if isinstance(item, str):
            return ('local', spec.owner, item)
        found = declarations.handles.get(item) if isinstance(item, Expression) else None
        if found is None:
            error_msg = (
                f'A noise law of {spec.owner} names the process {item!r}, which is not a label or the handle `dW` of a '
                '`WienerProcessSpec` bound to a role.'
            )
            raise ValidationError(error_msg)
        return ('wiener', *found)

    @staticmethod
    def _process_keys(spec: KernelSpec, law: NoiseLaw, declarations: Declarations) -> list[ProcessKey | None]:
        """Return the process of every component of the target of a noise law, `None` for an independent process."""
        count = len(components_of(law._get_targets()[0]))
        process = law.process
        if process is None:
            return [None] * count
        items = list(process) if isinstance(process, tuple) else components_of(process)
        if len(items) == 1:
            items = items * count
        if len(items) != count:
            error_msg = (
                f'A noise law of {spec.owner} names {len(items)} processes for a target of {count} components. Name '
                'one process, or one per component.'
            )
            raise ValidationError(error_msg)
        return [NoisePass._process_key(spec, item, declarations) for item in items]

    @staticmethod
    def _process_covariance(first: object, second: object, wieners: Mapping[str, WienerProcessSpec]) -> float:
        """Return the covariance per unit time of the increments of two processes, keys or independent markers."""
        if not (isinstance(first, tuple) and isinstance(second, tuple) and first[0] == 'wiener' == second[0]):
            return 1.0 if first == second else 0.0
        if first[1] != second[1]:
            return 0.0
        covariance = wieners[first[1]].covariance
        if covariance is None:
            return 1.0 if first[2] == second[2] else 0.0
        return float(np.asarray(covariance, dtype=float)[first[2], second[2]])
