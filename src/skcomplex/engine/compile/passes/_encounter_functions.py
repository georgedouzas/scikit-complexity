"""Pass that adds the gain and loss rates of encounter laws to the drift."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, cast

import numpy as np

from ....core.types import Vector
from ...problems._encounter_kernel import EncounterKernel
from ...problems._state import State
from .._base import BasePass
from .._declarations import Declarations


class EncounterFunctionsPass(BasePass):
    """Pass that adds the gain and loss rates of encounter laws to the drift."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('functions', 'encounters', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('functions',)

    def _run(self, inputs: object = None) -> None:
        """Add the rates of the encounter laws to the drift."""
        compilation = self.compilation_
        self._with_encounters(
            cast('dict[str, Any]', compilation.functions),
            compilation.encounters,
            cast('Declarations', compilation.layout).ends[True],
        )

    @staticmethod
    def _with_encounters(functions: dict[str, Any], encounters: tuple[EncounterKernel, ...], size: int) -> None:
        """Add the gain and loss rates of encounter laws to the drift, which then runs outside compiled steppers."""
        if not encounters:
            return
        drift = functions.get('drift')

        def encountered(t: float, x: Vector, p: Vector) -> Vector:
            rates = np.zeros(size) if drift is None else np.array(drift(t, x, p), dtype=float)
            for encounter in encounters:
                rates += encounter.run(State(t, x, p))
            return rates

        functions['drift'] = encountered
        functions['accelerated'] = None
