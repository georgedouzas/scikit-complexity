"""Climate variability.

The module covers stochastic climate and the delayed oscillator of El Niño.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws.utils import delayed
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations
from ..common.particles import RelaxationTimeInteraction


class ClimateAnomalyElement(Element):
    """Slow temperature anomaly of Hasselmann's stochastic climate.

    The anomaly has its feedback rate and weather noise.

    The relaxation of `HasselmannSystem` damps the anomaly at the rate `lam` and drives it with weather noise of
    amplitude `sigma`.

    Args:
        label:
            The label of the anomaly.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        T:
            The initial temperature anomaly.

        lam:
            The feedback rate that damps the anomaly.

        sigma:
            The amplitude of the weather noise.

    Examples:
        >>> from skcomplex.systems.physics.climate import ClimateAnomalyElement
        >>> ClimateAnomalyElement('ocean').lam
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        T: Annotated[float, StateSpec()] = 0.0,
        lam: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.T = T
        self.lam = lam
        self.sigma = sigma


class HasselmannSystem(BaseSystem):
    """Hasselmann's stochastic climate.

    Fast weather noise drives a slow, damped temperature anomaly.

    The anomaly follows `dT = -lam T dt + sigma dW`, an Ornstein-Uhlenbeck process whose red spectrum integrates white
    weather noise, with the stationary variance `sigma^2 / (2 lam)`. The system has the anomaly `ocean.T`.

    Args:
        T:
            The initial temperature anomaly.

        lam:
            The feedback rate that damps the anomaly.

        sigma:
            The amplitude of the weather noise.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.climate import HasselmannSystem
        >>> HasselmannSystem(lam=0.1).prepare().system_.elements[0].lam
        0.1
    """

    field = 'physics/climate'
    topics = ('sde', 'Ornstein-Uhlenbeck', 'red noise', 'stochastic climate')
    references = ('Hasselmann 1976', 'Ghil and Lucarini 2020')
    until = 100.0
    dt = 0.1
    solver = EulerMaruyamaSolver(dt=0.01)

    def __init__(
        self,
        T: float = 0.0,
        *,
        lam: float = 0.5,
        sigma: float = 0.3,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.T = T
        self.lam = lam
        self.sigma = sigma

    @staticmethod
    def _get_damping_time(s: SimpleNamespace) -> Expression:
        """Return the damping time `1 / lam` of the temperature anomaly."""
        return 1 / s.target.lam

    @staticmethod
    def _get_weather(s: SimpleNamespace) -> Expression:
        """Return the amplitude `sigma` of the weather noise."""
        return s.target.sigma

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        feedback = RelaxationTimeInteraction(
            'feedback',
            target='ocean',
            state='T',
            goal=0.0,
            time=self._get_damping_time,
            noise=self._get_weather,
        )
        return CompositeSystem(
            self.label,
            elements=[ClimateAnomalyElement('ocean', T=self.T, lam=self.lam, sigma=self.sigma)],
            interactions=[feedback],
        )


class ENSOSystem(DynamicalSystem):
    """The delayed oscillator of El Niño of Suarez and Schopf.

    The oscillator is `dT/dt = T - T^3 - alpha T(t - delta)`.

    The sea surface temperature anomaly of the eastern Pacific grows by local feedback and is damped by the delayed
    return of oceanic waves reflected at the western boundary. For `alpha < 1` and a long enough delay the warm and cold
    states lose stability to an oscillation between El Niño and La Niña. The history before the start is the initial
    anomaly. The system has the anomaly `pacific.T`.

    Args:
        T:
            The initial temperature anomaly, also its history.

        alpha:
            The strength of the delayed negative feedback.

        delta:
            The delay of the wave transit.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.climate import ENSOSystem
        >>> ENSOSystem(delta=4.0).prepare().system_.elements[0].get_value('delta')
        4.0
    """

    field = 'physics/climate'
    topics = ('delay', 'oscillation', 'El Niño', 'ENSO')
    references = ('Suarez and Schopf 1988', 'Battisti and Hirst 1989', 'Ghil and Lucarini 2020')
    until = 200.0
    dt = 0.5

    def __init__(
        self,
        T: float = 0.1,
        *,
        alpha: float = 0.75,
        delta: float = 6.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.T = T
        self.alpha = alpha
        self.delta = delta

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'pacific'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'T': self.T,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'alpha': self.alpha,
            'delta': self.delta,
        }

    def _get_equations(self) -> Equations:
        """Return the local growth and saturation of the anomaly, and its delayed feedback."""
        return {
            'T': lambda s: s.target.T - s.target.T**3 - s.target.alpha * delayed(s.target.T, s.target.delta),
        }
