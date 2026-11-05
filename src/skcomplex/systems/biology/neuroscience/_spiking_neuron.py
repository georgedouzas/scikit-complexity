"""Leaky integrate-and-fire neuron with an exponential synaptic trace of its own spikes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class SpikingNeuronElement(Element):
    """Leaky integrate-and-fire neuron with an exponential synaptic trace of its own spikes.

    The potential follows `tau v' = -v + mu` plus synaptic input, and the trace decays as `s' = -s / tau_s`. A spike
    reset sets the potential back and raises the trace by `1 / tau_s`, so every spike carries a unit area of trace.
    Units are mV and ms. The element holds the states and parameters and has no laws of its own: the leak and trace
    interactions of `delayed_lif_network` read its parameters.

    Args:
        label:
            The label of the neuron.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        v:
            The initial potential.

        s:
            The initial synaptic trace.

        tau:
            The membrane time constant.

        mu:
            The constant external drive, the potential the membrane relaxes to.

        tau_s:
            The decay time of the synaptic trace.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import SpikingNeuronElement
        >>> SpikingNeuronElement('neuron').tau_s
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[float, StateSpec()] = 0.0,
        s: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20.0,
        mu: Annotated[float, ParameterSpec()] = 25.0,
        tau_s: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v
        self.s = s
        self.tau = tau
        self.mu = mu
        self.tau_s = tau_s
