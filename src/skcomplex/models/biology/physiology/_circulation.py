"""Physiology models: lumped circulation of the blood."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Any

from sage.all__sagemath_symbolics import sin

from ....core.base import BaseInteraction, BaseLaw, Expression, NumericDomain, parameter, role, state
from ....core.components import Element
from ....core.laws import RateLaw, Symbols, mod, where
from ....core.system import System
from ....engine.runtime import register_model


def _pressure(chamber: Symbols, t: Expression) -> Expression:
    """Return the pressure of a chamber, its elastance times its volume above the unstressed volume."""
    phase = mod(t, chamber.period)
    activation = where(phase < chamber.systole, sin(math.pi * phase / chamber.systole), 0)
    return (chamber.E_min + (chamber.E_max - chamber.E_min) * activation) * (chamber.V - chamber.V0)


class ChamberElement(Element):
    """Elastic compartment of blood, a heart chamber or a vascular bed, whose pressure is elastance times volume.

    The pressure is `E(t) (V - V0)`. The elastance of a heart chamber rises from `E_min` to `E_max` and back as a half
    sine during the systole that starts every `period`, and the elastance of a vessel stays at `E_min` when `E_max`
    equals it. Volumes are in ml, pressures in mmHg, and times in s.

    Args:
        label:
            The label of the chamber.

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
        >>> from skcomplex.models.biology.physiology import ChamberElement
        >>> ChamberElement('arteries', V=150.0, E_min=0.8, E_max=0.8).E_max
        0.8
    """

    V: float = state(default=100.0, domain=NumericDomain.NONNEGATIVE)
    V0: float = parameter(default=0.0, domain=NumericDomain.NONNEGATIVE)
    E_min: float = parameter(default=1.0, domain=NumericDomain.POSITIVE)
    E_max: float = parameter(default=1.0, domain=NumericDomain.POSITIVE)
    period: float = parameter(default=0.8, domain=NumericDomain.POSITIVE)
    systole: float = parameter(default=0.3, domain=NumericDomain.POSITIVE)


class TransferInteraction(BaseInteraction):
    """Transfer of blood from one chamber to another through a resistance, `Q = (P_up - P_down) / R`.

    A valve lets blood pass forward only, so the flow is zero when the downstream pressure is higher.

    Args:
        label:
            The label of the interaction.

        valve:
            Whether a valve prevents backward flow.

        **values:
            The values of the declared variables, by name:

            upstream:
                The path of the chamber the blood leaves.

            downstream:
                The path of the chamber the blood enters.

            R:
                The resistance in mmHg s/ml.

    Examples:
        >>> from skcomplex.models.biology.physiology import TransferInteraction
        >>> TransferInteraction('aortic_valve', upstream='ventricle', downstream='arteries', valve=True).valve
        True
    """

    upstream: str = role()
    downstream: str = role()
    R: float = parameter(default=1.0, domain=NumericDomain.POSITIVE)

    def __init__(
        self,
        label: str,
        *,
        valve: bool = False,
        **values: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(label, **values)
        self.valve = valve

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the transfer of volume from the upstream to the downstream chamber.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        drop = _pressure(s.upstream, s.t) - _pressure(s.downstream, s.t)
        flow = where(drop > 0, drop / s.R, 0) if self.valve else drop / s.R
        return [RateLaw(s.upstream.V, -flow), RateLaw(s.downstream.V, flow)]


@register_model(
    field='biology/physiology',
    tags=('cardiovascular', 'circulation', 'lumped model', 'hybrid'),
    references=('Suga and Sagawa 1974', 'Trzaska 2023'),
    defaults={'until': 8.0, 'dt': 0.02},
)
def cardiovascular(
    *,
    heart_rate: float = 75.0,
    E_max: float = 2.5,
    R_systemic: float = 1.0,
    C_arterial: float = 1.3,
    C_venous: float = 50.0,
    volume: float = 1500.0,
) -> System:
    """Return a closed lumped model of the systemic circulation driven by a left ventricle of varying elastance.

    Blood leaves the ventricle through the aortic valve into the compliant arteries, flows through the systemic
    resistance into the veins, and returns through the mitral valve. The ventricle stiffens in systole, so the model
    beats, the valves open and close, and the total volume of blood stays constant. With the defaults the arterial
    pressure swings between about 80 and 120 mmHg.

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

    Returns:
        system:
            The model, with the volumes `ventricle.V`, `arteries.V`, and `veins.V`.

    Examples:
        >>> from skcomplex.models.biology.physiology import cardiovascular
        >>> sum(chamber.V for chamber in cardiovascular().elements)
        1500.0
    """
    period = 60.0 / heart_rate
    systole = 0.375 * period
    ventricle = ChamberElement(
        'ventricle',
        V=0.1 * volume,
        V0=10.0,
        E_min=0.08,
        E_max=E_max,
        period=period,
        systole=systole,
    )
    arteries = ChamberElement(
        'arteries',
        V=0.3 * volume,
        V0=250.0,
        E_min=1 / C_arterial,
        E_max=1 / C_arterial,
        period=period,
        systole=systole,
    )
    veins = ChamberElement(
        'veins',
        V=0.6 * volume,
        V0=500.0,
        E_min=1 / C_venous,
        E_max=1 / C_venous,
        period=period,
        systole=systole,
    )
    flows = [
        TransferInteraction('aortic_valve', upstream='ventricle', downstream='arteries', R=0.02, valve=True),
        TransferInteraction('systemic', upstream='arteries', downstream='veins', R=R_systemic),
        TransferInteraction('mitral_valve', upstream='veins', downstream='ventricle', R=0.02, valve=True),
    ]
    return System(elements=[ventricle, arteries, veins], interactions=flows)
