"""Cell of an automaton over a ring of formal power series truncated at an order."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

import numpy as np

from ....core.components.elements import DeclaredElement
from ....core.specs import DeclaredVariableSpec, PowerSeriesSpec, StateSpec, VisualSpec

RATIONAL_SERIES = PowerSeriesSpec()


class PowerSeriesCellElement(DeclaredElement):
    """Cell of an automaton over a ring `K[[t]]` of formal power series truncated at an order.

    The cell declares the state `value` in the ring it is given, so it holds the coefficients `a_0, ..., a_{order - 1}`
    of a series, which a local rule reads as one row per cell.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        ring:
            The truncated power series ring of the values.

        variables:
            Declared states and parameters besides `value`, or values that replace its default, by name.

    Examples:
        >>> from skcomplex.core.specs import PowerSeriesSpec
        >>> from skcomplex.systems.computer_science.automata import PowerSeriesCellElement
        >>> PowerSeriesCellElement('cell', ring=PowerSeriesSpec(3)).get_value('value').tolist()
        [0.0, 0.0, 0.0]
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        ring: PowerSeriesSpec = RATIONAL_SERIES,
        variables: Mapping[str, DeclaredVariableSpec] | None = None,
    ) -> None:
        super().__init__(label, visual=visual, variables=variables)
        self.ring = ring

    def _declare_variables(self) -> Mapping[str, DeclaredVariableSpec]:
        """Return the state `value` in the ring, with the zero series as its default, and the given variables."""
        value = DeclaredVariableSpec(np.zeros(self.ring.order), StateSpec(domain=self.ring))
        return {'value': value, **(self.variables or {})}
