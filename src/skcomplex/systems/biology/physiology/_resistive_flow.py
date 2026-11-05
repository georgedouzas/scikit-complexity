"""Flow of blood from one chamber to another through a resistance, `Q = (P_source - P_target) / R`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo, sin
from sage.sets.real_set import RealSet

from ....core.components.interactions import TransferInteraction
from ....core.components.laws.utils import mod, where
from ....core.specs import ParameterSpec, RoleSpec


class ResistiveFlowInteraction(TransferInteraction):
    """Flow of blood from one chamber to another through a resistance, `Q = (P_source - P_target) / R`.

    A valve lets blood pass forward only, so the flow is zero when the target pressure is higher.

    Args:
        label:
            The label of the interaction.

        source:
            The path of the chamber the blood leaves.

        target:
            The path of the chamber the blood enters.

        R:
            The resistance in mmHg s/ml.

        valve:
            Whether a valve prevents backward flow.

    Examples:
        >>> from skcomplex.systems.biology.physiology import ResistiveFlowInteraction
        >>> ResistiveFlowInteraction('aortic_valve', source='ventricle', target='arteries', valve=True).valve
        True
    """

    def __init__(
        self,
        label: str,
        *,
        source: Annotated[str, RoleSpec()],
        target: Annotated[str, RoleSpec()],
        R: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        valve: bool = False,
    ) -> None:
        super().__init__(label, source=source, target=target, state='V', amount=self._get_flow)
        self.R = R
        self.valve = valve

    def _get_pressure(self, chamber: SimpleNamespace, t: Expression) -> Expression:
        """Return the pressure of a chamber, its elastance times its volume above the unstressed volume."""
        phase = mod(t, chamber.period)
        activation = where(phase < chamber.systole, sin(math.pi * phase / chamber.systole), 0)
        return (chamber.E_min + (chamber.E_max - chamber.E_min) * activation) * (chamber.V - chamber.V0)

    def _get_flow(self, s: SimpleNamespace) -> Expression:
        """Return the flow of the pressure drop through the resistance, zero backwards through a valve."""
        drop = self._get_pressure(s.source, s.t) - self._get_pressure(s.target, s.t)
        return where(drop > 0, drop / s.R, 0) if self.valve else drop / s.R
