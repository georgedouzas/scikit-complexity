"""Define the composed rule spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import IntArray
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class _ComposedRuleSpec(BaseSpec):
    """Rule of the composition of two automata, on the composite neighbourhood."""

    first: IntArray
    second: IntArray
    columns: tuple[tuple[int, ...], ...]
    states: int

    def evaluate(self, *values: npt.NDArray[Any]) -> npt.NDArray[np.int64]:
        """Return the value of the second rule on the values of the first rule around every cell."""
        from ..analysis._automata import AutomatonAnalysis
        cells = np.stack([np.asarray(value, dtype=np.int64) for value in values], axis=-1)
        inner = [
            self.first[AutomatonAnalysis._codes(cells[..., list(columns)], self.states)] for columns in self.columns
        ]
        return self.second[AutomatonAnalysis._codes(np.stack(inner, axis=-1), self.states)]
