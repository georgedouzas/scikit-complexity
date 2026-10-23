"""Add to the drift the correction of every noise law of a convention other than Itô."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from sage.all__sagemath_symbolics import SR

from ...base import Expression, SymbolicVector, ValidationError, components_of
from ...components import Wiener
from ...laws import Law, Noise, Rate
from ...laws._laws import NOISE_CONVENTIONS
from .._compilation import Compilation, Laws
from .._declarations import Declarations
from .._noise import process_covariance, process_keys
from .._pipeline import Pass
from .._specs import KernelSpec


def _noise_terms(specs: list[tuple[KernelSpec, list[Law]]], declarations: Declarations) -> list[_NoiseTerm]:
    """Return every component of every noise law, an independent process standing for itself alone."""
    terms = []
    for spec, laws in specs:
        for law in laws:
            if not isinstance(law, Noise):
                continue
            targets = components_of(law.target)
            values = (
                components_of(law.value)
                if isinstance(law.value, SymbolicVector | list | tuple)
                else [law.value] * len(targets)
            )
            keys = process_keys(spec, law, declarations)
            for index, (target, value, key) in enumerate(zip(targets, values, keys, strict=True)):
                reference = spec.references[target]
                process = key if key is not None else ('independent', id(law), index)
                terms.append(_NoiseTerm(spec, law, target, (reference.entry.name, reference.column), value, process))
    return terms


def _reads_state(term: _NoiseTerm, state: tuple[str, int]) -> bool:
    """Check whether the amplitude of a noise term depends on a state component."""
    amplitude = SR(term.amplitude)
    return any(
        (reference.entry.name, reference.column) == state and not amplitude.diff(item).is_zero()
        for item in amplitude.variables()
        if (reference := term.spec.references.get(item)) is not None and reference.is_state
    )


def _noise_correction(term: _NoiseTerm, terms: list[_NoiseTerm], wieners: Mapping[str, Wiener]) -> Law | None:
    """Return the drift correction `alpha sum C_jl G_kl dG_ij/dx_k` of a noise term of a convention, or `None`."""
    alpha = NOISE_CONVENTIONS[term.law.convention]
    if not alpha or not isinstance(term.amplitude, Expression | int | float):
        return None
    amplitude = SR(term.amplitude)
    parts = []
    for other in terms:
        covariance = process_covariance(term.process, other.process, wieners)
        if not covariance:
            continue
        if other.spec is not term.spec:
            if _reads_state(term, other.state):
                error_msg = (
                    f'The {term.law.convention} noise law of {term.spec.owner} depends on {other.state[0]}, which a '
                    f'noise law of {other.spec.owner} drives through a shared process, so its drift correction is out '
                    'of scope. Use the Itô convention, or write both laws in one component.'
                )
                raise ValidationError(error_msg)
            continue
        derivative = amplitude.diff(other.target)
        if derivative.is_zero():
            continue
        if not isinstance(other.amplitude, Expression | int | float):
            error_msg = (
                f'The {term.law.convention} noise law of {term.spec.owner} shares a process with a noise law whose '
                'amplitude is not symbolic, so its drift correction is unknown. Write the amplitude as an expression.'
            )
            raise ValidationError(error_msg)
        parts.append(alpha * covariance * SR(other.amplitude) * derivative)
    if not parts:
        return None
    correction = sum(parts[1:], parts[0])
    return None if correction.is_zero() else Rate(term.target, correction)


def _with_noise_corrections(
    specs: list[tuple[KernelSpec, list[Law]]],
    declarations: Declarations,
) -> list[tuple[KernelSpec, list[Law]]]:
    """Return laws with the drift correction of every noise law of a convention after it, in the Itô form."""
    terms = _noise_terms(specs, declarations)
    by_law: dict[int, list[_NoiseTerm]] = defaultdict(list)
    for term in terms:
        by_law[id(term.law)].append(term)
    result = []
    for spec, laws in specs:
        corrected: list[Law] = []
        for law in laws:
            corrected.append(law)
            for term in by_law.get(id(law), []):
                correction = _noise_correction(term, terms, declarations.wieners)
                if correction is not None:
                    corrected.append(correction)
        result.append((spec, corrected))
    return result


@dataclass(frozen=True)
class _NoiseTerm:
    """Amplitude of a noise law on one state component, with the process it draws."""

    spec: KernelSpec
    law: Noise
    target: Expression
    state: tuple[str, int]
    amplitude: object
    process: object


class NoisePass(Pass):
    """Pass that adds the drift correction of every noise law of a convention after it, in the Itô form."""

    reads = ('delayed', 'layout')
    writes = ('corrected',)

    def run(self, compilation: Compilation) -> None:
        """Add the drift corrections of the noise laws.

        Args:
            compilation:
                The compilation state.
        """
        compilation.corrected = _with_noise_corrections(
            cast('Laws', compilation.delayed),
            cast('Declarations', compilation.layout),
        )
