"""Electrode of a stirred electrochemical reactor, charged by a train of current pulses."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import pulse_train
from ....specs import ParameterSpec, StateSpec, VisualSpec


class PulsedElectrodeElement(Element):
    """Electrode of a stirred electrochemical reactor, charged by a train of current pulses.

    In units of the exchange current and of the thermal voltage, the overpotential follows
    `C deta/dt = i(t) - i_F`, with the Butler-Volmer current `i_F = c exp(alpha eta) - exp(-(1 - alpha) eta)`, and the
    reactant concentration follows `dc/dt = (1 - c) / tau - kappa i_F`.

    Args:
        label:
            The label of the electrode.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        eta:
            The initial overpotential.

        c:
            The initial reactant concentration, relative to the feed.

        C:
            The capacitance of the double layer.

        alpha:
            The transfer coefficient.

        tau:
            The residence time of the reactor.

        kappa:
            The consumption of reactant per unit of Faradaic current.

        current:
            The current of a pulse.

        period:
            The time between the starts of pulses.

        width:
            The duration of a pulse.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import PulsedElectrodeElement
        >>> PulsedElectrodeElement('electrode').period
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        eta: Annotated[float, StateSpec()] = 0.0,
        c: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        C: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        kappa: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        current: Annotated[float, ParameterSpec()] = 5.0,
        period: float = 10.0,
        width: float = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.eta = eta
        self.c = c
        self.C = C
        self.alpha = alpha
        self.tau = tau
        self.kappa = kappa
        self.current = current
        self.period = period
        self.width = width

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the charging of the double layer and the balance of the reactant.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        faradaic = s.c * exp(s.alpha * s.eta) - exp(-(1 - s.alpha) * s.eta)
        applied = s.current * pulse_train(s.t, self.period, self.width)
        return [
            RateLaw(s.eta, (applied - faradaic) / s.C),
            RateLaw(s.c, (1 - s.c) / s.tau - s.kappa * faradaic),
        ]
