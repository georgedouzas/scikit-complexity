"""Pass that adds the gain and loss rates of encounter laws to the drift."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, cast

import numpy as np

from ....core.types import Vector
from .._compilation import Compilation
from .._compiled_encounter import CompiledEncounter
from .._declarations import Declarations
from .._pipeline import BasePass


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
