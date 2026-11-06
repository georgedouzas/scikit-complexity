"""Population of quadratic integrate-and-fire neurons in the exact mean field of Montbrió, Pazó, and Roxin."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, pi
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class NeuralMassElement(Element):
    """Population of quadratic integrate-and-fire neurons in the exact mean field of Montbrió, Pazó, and Roxin.

    The excitabilities of the neurons follow a Lorentzian distribution of median `H` and half-width `delta`, so the
    firing rate `r` and the mean membrane potential `v` of the population follow `dr/dt = delta / (tau^2 pi) + 2 r v
    / tau` and `dv/dt = (v^2 + H) / tau - tau (pi r)^2` exactly, with the membrane time constant `tau`. The synaptic
    field `s` filters the rate with exponential synapses, `ds/dt = (r - s) / tau_d`. Couplings and inputs add to the
    rate of `v`.

    Args:
        label:
            The label of the population.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        r:
            The initial firing rate, in spikes per millisecond.

        v:
            The initial mean membrane potential.

        s:
            The initial synaptic field.

        tau:
            The membrane time constant, in milliseconds.

        delta:
            The half-width of the Lorentzian excitabilities.

        H:
            The median of the Lorentzian excitabilities.

        tau_d:
            The decay time of the synapses, in milliseconds.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import NeuralMassElement
        >>> NeuralMassElement('excitatory', tau=20.0).tau
        20.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        r: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        v: Annotated[float, StateSpec()] = -1.0,
        s: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20.0,
        delta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        H: Annotated[float, ParameterSpec()] = -5.0,
        tau_d: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.r = r
        self.v = v
        self.s = s
        self.tau = tau
        self.delta = delta
        self.H = H
        self.tau_d = tau_d

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the mean-field rate, potential, and synaptic field.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            RateLaw(s.r, s.delta / (s.tau**2 * pi) + 2 * s.r * s.v / s.tau),
            RateLaw(s.v, (s.v**2 + s.H) / s.tau - s.tau * (pi * s.r) ** 2),
            RateLaw(s.s, (s.r - s.s) / s.tau_d),
        ]
