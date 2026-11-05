"""Rate neuron that adapts its gain and threshold so that its firing rate becomes exponentially distributed."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class PolyhomeostaticNeuronElement(Element):
    """Rate neuron that adapts its gain and threshold so that its firing rate becomes exponentially distributed.

    The membrane potential `x` integrates the input `I0 + A sin(omega t)` with rate `Gamma`, and the neuron fires at
    rate `y = 1 / (1 + exp(a (b - x)))`. Stochastic gradient descent on the Kullback-Leibler divergence between the
    distribution of `y` and an exponential of mean `mu` gives `a' = eps_a (1/a + (x - b) d)` and `b' = -eps_b a d`
    with `d = 1 - (2 + 1/mu) y + y²/mu`. The element holds the states and parameters and has no laws of its own: the
    relaxation and plasticity interactions of `polyhomeostatic_neuron` read its parameters.

    Args:
        label:
            The label of the neuron.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial membrane potential.

        a:
            The initial gain.

        b:
            The initial threshold.

        mu:
            The target mean firing rate.

        eps_a:
            The adaptation rate of the gain.

        eps_b:
            The adaptation rate of the threshold.

        Gamma:
            The leak rate of the membrane potential.

        I0:
            The mean input.

        A:
            The amplitude of the periodic input.

        omega:
            The angular frequency of the periodic input.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import PolyhomeostaticNeuronElement
        >>> PolyhomeostaticNeuronElement('neuron').mu
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        a: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        b: Annotated[float, StateSpec()] = 0.0,
        mu: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.2,
        eps_a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        eps_b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        Gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        I0: Annotated[float, ParameterSpec()] = 0.0,
        A: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        omega: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.a = a
        self.b = b
        self.mu = mu
        self.eps_a = eps_a
        self.eps_b = eps_b
        self.Gamma = Gamma
        self.I0 = I0
        self.A = A
        self.omega = omega
