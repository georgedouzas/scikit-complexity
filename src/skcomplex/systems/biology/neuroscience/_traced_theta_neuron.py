"""Theta neuron that keeps traces of its own spikes, which spike-timing-dependent plasticity reads."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, oo, pi
from sage.sets.real_set import RealSet

from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._theta_neuron import ThetaNeuronElement


class TracedThetaNeuronElement(ThetaNeuronElement):
    """Theta neuron that keeps traces of its own spikes, which spike-timing-dependent plasticity reads.

    A spike is the smooth pulse `P(theta) = (1 - cos theta)^2 * 2 / 3`, of unit mean over the phase, which peaks as the
    phase passes π. The presynaptic trace `a` and the postsynaptic trace `b` rise with every pulse and decay with the
    times `tau_plus` and `tau_minus`, `da/dt = P(theta) - a / tau_plus` and `db/dt = P(theta) - b / tau_minus`, so
    they hold the recent spikes weighted by the exponential windows of the plasticity.

    Args:
        label:
            The label of the neuron or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        theta:
            The initial phase.

        a:
            The initial presynaptic trace.

        b:
            The initial postsynaptic trace.

        eta:
            The excitability.

        tau_plus:
            The time of the potentiation window.

        tau_minus:
            The time of the depression window.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import TracedThetaNeuronElement
        >>> TracedThetaNeuronElement('neuron').tau_plus
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
        a: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        b: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        eta: Annotated[float, ParameterSpec()] = 1.0,
        tau_plus: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        tau_minus: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual, theta=theta, eta=eta)
        self.a = a
        self.b = b
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the phase dynamics and the traces of the spikes.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        pulse = 2 * (1 - cos(s.theta)) ** 2 / 3
        return [
            *super()._get_laws(s),
            RateLaw(s.a, pulse - s.a / s.tau_plus),
            RateLaw(s.b, pulse - s.b / s.tau_minus),
        ]
