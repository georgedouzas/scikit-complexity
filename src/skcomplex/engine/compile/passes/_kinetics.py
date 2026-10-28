"""Compile kinetic encounter laws, whose gain and loss rates add to the drift of the activity levels."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field, fields
from typing import Any, cast

import numpy as np

from ....core.types import Vector
from ...backends import BaseBackend, resolve_backend
from ...problems import Problem
from .._compilation import Compilation, Laws
from .._declarations import Declarations, _member_count
from .._kinetic_contractions import CompiledEncounter
from .._kinetic_encounters import _KINETIC_LAWS, _compile_encounters
from .._pipeline import BasePass


@dataclass(frozen=True)
class KineticProblem(Problem):
    """Problem whose drift holds the gain and loss rates of encounter laws, with the compiled encounters.

    Args:
        encounters:
            The encounter laws, whose gain and loss rates the drift holds and particle solvers sample.
    """

    encounters: tuple[CompiledEncounter, ...] = field(default=(), repr=False)


class CompileEncountersPass(BasePass):
    """Pass that compiles the encounter laws before reductions prune the groups they read, and takes them out."""

    reads = ('drawn', 'layout')
    writes = ('drawn', 'encounters')

    def _run(self, compilation: Compilation) -> None:
        """Compile the encounter laws of the drawn laws.

        Args:
            compilation:
                The compilation state.
        """
        specs = cast('Laws', compilation.drawn)
        declarations = cast('Declarations', compilation.layout)
        compilation.facts['encounters'] = self._compiled_encounters(specs, declarations, compilation.backend)
        compilation.drawn = [
            (spec, [law for law in laws if not isinstance(law, _KINETIC_LAWS)]) for spec, laws in specs
        ]

    @staticmethod
    def _compiled_encounters(
        specs: Laws,
        declarations: Declarations,
        backend: BaseBackend | None,
    ) -> tuple[CompiledEncounter, ...]:
        """Return the encounter laws of every component, lowered for their members."""
        if not any(isinstance(law, _KINETIC_LAWS) for _, laws in specs for law in laws):
            return ()
        members = _member_count(list(declarations.blocks.values()))
        fused = resolve_backend(backend, members).name == 'numba'
        components = [(spec.owner, dict(spec.references), laws) for spec, laws in specs]
        return _compile_encounters(components, declarations.time, fused=fused)


class EncounterFunctionsPass(BasePass):
    """Pass that adds the gain and loss rates of encounter laws to the drift."""

    reads = ('functions', 'encounters', 'layout')
    writes = ('functions',)

    def _run(self, compilation: Compilation) -> None:
        """Add the rates of the encounter laws to the drift.

        Args:
            compilation:
                The compilation state.
        """
        self._with_encounters(
            cast('dict[str, Any]', compilation.functions),
            compilation.facts['encounters'],
            cast('Declarations', compilation.layout).ends[True],
        )

    @staticmethod
    def _with_encounters(functions: dict[str, Any], encounters: tuple[CompiledEncounter, ...], size: int) -> None:
        """Add the gain and loss rates of encounter laws to the drift, which then runs outside compiled steppers."""
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


class EncounterJacobianPass(BasePass):
    """Pass that drops the exact Jacobian of a problem with encounter laws, whose rates it does not derive."""

    reads = ('jacobian', 'encounters')
    writes = ('jacobian',)

    def _run(self, compilation: Compilation) -> None:
        """Drop the Jacobian of a problem with encounter laws.

        Args:
            compilation:
                The compilation state.
        """
        if compilation.facts['encounters']:
            compilation.jacobian = None


class EncounterProblemPass(BasePass):
    """Pass that makes the problem of a system with encounter laws a kinetic problem that holds them."""

    reads = ('problem', 'encounters')
    writes = ('problem',)

    def _run(self, compilation: Compilation) -> None:
        """Add the encounter laws to the problem.

        Args:
            compilation:
                The compilation state.
        """
        encounters = compilation.facts['encounters']
        if not encounters:
            return
        problem = cast('Problem', compilation.problem)
        values = {item.name: getattr(problem, item.name) for item in fields(problem)}
        compilation.problem = KineticProblem(**values, encounters=encounters)
