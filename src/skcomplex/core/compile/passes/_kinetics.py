"""Compile kinetic encounter laws, whose gain and loss rates add to the drift of the activity levels."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace
from typing import Any, cast

import numpy as np

from .._backends import Backend, resolve_backend
from .._compilation import Compilation, Laws
from .._contractions import CompiledEncounter
from .._declarations import Declarations, member_count
from .._encounters import ENCOUNTER_LAWS, compile_encounters
from .._pipeline import Pass
from .._problem import Problem, Vector


def _compiled_encounters(
    specs: Laws,
    declarations: Declarations,
    backend: str | Backend,
) -> tuple[CompiledEncounter, ...]:
    """Return the encounter laws of every component, lowered for their members."""
    if not any(isinstance(law, ENCOUNTER_LAWS) for _, laws in specs for law in laws):
        return ()
    members = member_count(list(declarations.blocks.values()))
    fused = resolve_backend(backend, members).name == 'numba'
    components = [(spec.owner, dict(spec.references), laws) for spec, laws in specs]
    return compile_encounters(components, declarations.time, fused=fused)


def _with_encounters(functions: dict[str, Any], encounters: tuple[CompiledEncounter, ...], size: int) -> None:
    """Add the gain and loss rates of encounter laws to the drift, which then runs outside compiled stepper loops."""
    if not encounters:
        return
    drift = functions.get('drift')

    def encountered(t: float, x: Vector, p: Vector) -> Vector:
        rates = np.zeros(size) if drift is None else np.array(drift(t, x, p), dtype=float)
        for encounter in encounters:
            rates += encounter.rates(x, p)
        return rates

    functions['drift'] = encountered
    functions['accelerated'] = None


class CompileEncountersPass(Pass):
    """Pass that compiles the encounter laws before reductions prune the groups they read, and takes them out."""

    reads = ('drawn', 'layout')
    writes = ('drawn', 'encounters')

    def run(self, compilation: Compilation) -> None:
        """Compile the encounter laws of the drawn laws.

        Args:
            compilation:
                The compilation state.
        """
        specs = cast('Laws', compilation.drawn)
        declarations = cast('Declarations', compilation.layout)
        compilation.facts['encounters'] = _compiled_encounters(specs, declarations, compilation.backend)
        compilation.drawn = [
            (spec, [law for law in laws if not isinstance(law, ENCOUNTER_LAWS)]) for spec, laws in specs
        ]


class EncounterFunctionsPass(Pass):
    """Pass that adds the gain and loss rates of encounter laws to the drift."""

    reads = ('functions', 'encounters', 'layout')
    writes = ('functions',)

    def run(self, compilation: Compilation) -> None:
        """Add the rates of the encounter laws to the drift.

        Args:
            compilation:
                The compilation state.
        """
        _with_encounters(
            cast('dict[str, Any]', compilation.functions),
            compilation.facts['encounters'],
            cast('Declarations', compilation.layout).ends[True],
        )


class EncounterJacobianPass(Pass):
    """Pass that drops the exact Jacobian of a problem with encounter laws, whose rates it does not derive."""

    reads = ('jacobian', 'encounters')
    writes = ('jacobian',)

    def run(self, compilation: Compilation) -> None:
        """Drop the Jacobian of a problem with encounter laws.

        Args:
            compilation:
                The compilation state.
        """
        if compilation.facts['encounters']:
            compilation.jacobian = None


class EncounterProblemPass(Pass):
    """Pass that adds the compiled encounter laws to the problem."""

    reads = ('problem', 'encounters')
    writes = ('problem',)

    def run(self, compilation: Compilation) -> None:
        """Add the encounter laws to the problem.

        Args:
            compilation:
                The compilation state.
        """
        compilation.problem = replace(cast('Problem', compilation.problem), encounters=compilation.facts['encounters'])


PASSES = (CompileEncountersPass(), EncounterFunctionsPass(), EncounterJacobianPass(), EncounterProblemPass())
