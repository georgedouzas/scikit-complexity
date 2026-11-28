"""Add the gains and losses of encounter laws to the rates."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace

import numpy as np

from ....errors import PipelineError
from ....types import Array, VectorFunction
from ...problems import RunStateSpec
from ...problems._encounter_kernel import EncounterKernel
from .._base import BasePass
from .._compilation import Compilation, _ProblemFunctions


class EncounterFunctionsPass(BasePass):
    """A pass that adds the gains and losses of encounter laws to the rates.

    Every encounter moves density from one class to another, and the pass adds the gain of the new class and the loss of
    the old one to the rates that the solver integrates.

    Read more in the user guide, under Extend the library.

    See Also:
        CompileEncountersPass: Compiles the encounters.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.social.opinion_dynamics import KineticOpinionSystem
        >>> compilation = PassCompiler().prepare(KineticOpinionSystem()).run().compilation_
        >>> list(compilation.functions.equations)
        ['opinions.x[i]']
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('functions', 'encounters', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('functions',)

    def _run(self, inputs: object = None) -> None:
        """Add the rates of the encounter laws to the drift."""
        compilation = self.compilation_
        functions, encounters = compilation.functions, compilation.encounters
        if functions is None or encounters is None:
            error_msg = 'EncounterFunctionsPass reads the numeric functions and the encounters, which no pass wrote.'
            raise PipelineError(error_msg)
        size = Compilation._get_required(compilation.layout, 'layout').ends[True]
        compilation.functions = self._copy_with_encounters(functions, encounters, size)

    @staticmethod
    def _copy_with_encounters(
        functions: _ProblemFunctions,
        encounters: tuple[EncounterKernel, ...],
        size: int,
    ) -> _ProblemFunctions:
        """Add the gain and loss rates of encounter laws to the drift.

        The drift then runs outside compiled steppers.
        """
        if not encounters:
            return functions

        def encountered_of(base: VectorFunction | None) -> VectorFunction:
            def encountered(t: float, x: Array, p: Array) -> Array:
                rates = np.zeros(size) if base is None else np.array(base(t, x, p), dtype=float)
                for encounter in encounters:
                    rates += np.asarray(encounter.run(RunStateSpec(t, x, p)).values_, dtype=float)
                return rates

            return encountered

        sources = None if functions.sources is None else encountered_of(functions.sources)
        return replace(functions, drift=encountered_of(functions.drift), sources=sources, accelerated=None)
