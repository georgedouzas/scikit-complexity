"""Physiology systems: lumped circulation of the blood."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import TransferInteraction
from ....core.components.laws.utils import mod, where
from ....core.components.system import BaseSystem, System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


class ChamberElement(Element):
    """Elastic compartment of blood, a heart chamber or a vascular bed, whose pressure is elastance times volume.

    The pressure is `E(t) (V - V0)`. The elastance of a heart chamber rises from `E_min` to `E_max` and back as a half
    sine during the systole that starts every `period`, and the elastance of a vessel stays at `E_min` when `E_max`
    equals it. Volumes are in ml, pressures in mmHg, and times in s.

    Args:
        label:
            The label of the chamber.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        V:
            The initial volume.

        V0:
            The unstressed volume.

        E_min:
            The elastance in diastole, the inverse of the compliance.

        E_max:
            The largest elastance in systole.

        period:
            The duration of a heartbeat.

        systole:
            The duration of the contraction.

    Examples:
        >>> from skcomplex.systems.biology.physiology import ChamberElement
        >>> ChamberElement('arteries', V=150.0, E_min=0.8, E_max=0.8).E_max
        0.8
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        V: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        V0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        E_min: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        E_max: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        period: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.8,
        systole: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.V = V
        self.V0 = V0
        self.E_min = E_min
        self.E_max = E_max
        self.period = period
        self.systole = systole


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


class CardiovascularSystem(BaseSystem):
    """A closed lumped system of the systemic circulation driven by a left ventricle of varying elastance.

    Blood leaves the ventricle through the aortic valve into the compliant arteries, flows through the systemic
    resistance into the veins, and returns through the mitral valve. The ventricle stiffens in systole, so the system
    beats, the valves open and close, and the total volume of blood stays constant. With the defaults the arterial
    pressure swings between about 80 and 120 mmHg. The system has the volumes `ventricle.V`, `arteries.V`, and
    `veins.V`.

    Args:
        heart_rate:
            The number of beats per minute.

        E_max:
            The largest elastance of the ventricle in mmHg/ml.

        R_systemic:
            The systemic resistance in mmHg s/ml.

        C_arterial:
            The arterial compliance in ml/mmHg.

        C_venous:
            The venous compliance in ml/mmHg.

        volume:
            The total volume of blood in ml, split at the start among the ventricle, arteries, and veins.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.physiology import CardiovascularSystem
        >>> sum(chamber.V for chamber in CardiovascularSystem().prepare().system_.elements)
        1500.0
    """

    field = 'biology/physiology'
    topics = ('cardiovascular', 'circulation', 'lumped system', 'hybrid')
    references = ('Suga and Sagawa 1974', 'Trzaska 2023')
    until = 8.0
    dt = 0.02

    def __init__(
        self,
        *,
        heart_rate: float = 75.0,
        E_max: float = 2.5,
        R_systemic: float = 1.0,
        C_arterial: float = 1.3,
        C_venous: float = 50.0,
        volume: float = 1500.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.heart_rate = heart_rate
        self.E_max = E_max
        self.R_systemic = R_systemic
        self.C_arterial = C_arterial
        self.C_venous = C_venous
        self.volume = volume

    def _build_system(self) -> System:
        """Return the parts of the system."""
        period = 60.0 / self.heart_rate
        systole = 0.375 * period
        ventricle = ChamberElement(
            'ventricle',
            V=0.1 * self.volume,
            V0=10.0,
            E_min=0.08,
            E_max=self.E_max,
            period=period,
            systole=systole,
        )
        arteries = ChamberElement(
            'arteries',
            V=0.3 * self.volume,
            V0=250.0,
            E_min=1 / self.C_arterial,
            E_max=1 / self.C_arterial,
            period=period,
            systole=systole,
        )
        veins = ChamberElement(
            'veins',
            V=0.6 * self.volume,
            V0=500.0,
            E_min=1 / self.C_venous,
            E_max=1 / self.C_venous,
            period=period,
            systole=systole,
        )
        flows = [
            ResistiveFlowInteraction('aortic_valve', source='ventricle', target='arteries', R=0.02, valve=True),
            ResistiveFlowInteraction('systemic', source='arteries', target='veins', R=self.R_systemic),
            ResistiveFlowInteraction('mitral_valve', source='veins', target='ventricle', R=0.02, valve=True),
        ]
        return System(self.label, elements=[ventricle, arteries, veins], interactions=flows)
