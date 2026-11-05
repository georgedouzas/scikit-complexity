"""Global coupling of theta neurons through the mean of the smooth pulses that every neuron emits."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import mean
from ....core.specs import ParameterSpec, RoleSpec


class PulseCouplingInteraction(Interaction):
    """Global coupling of theta neurons through the mean of the smooth pulses that every neuron emits.

    Every neuron receives the current `(1 + cos theta) kappa I`, where `I = mean_j a_n (1 - cos theta_j)^n` is the mean
    pulse of all neurons at the current state. The pulse peaks when a neuron fires, and `a_n` gives it a unit mean over
    the phase. With `kappa > 0` the coupling is excitatory.

    Args:
        label:
            The label of the interaction.

        neurons:
            The path of the group of theta neurons.

        kappa:
            The coupling strength.

        n:
            The sharpness of the pulses, a positive integer.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import PulseCouplingInteraction
        >>> PulseCouplingInteraction('coupling', neurons='neurons', kappa=2.0).kappa
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        neurons: Annotated[str, RoleSpec()],
        kappa: Annotated[float, ParameterSpec()] = 1.0,
        n: int = 2,
    ) -> None:
        super().__init__(label)
        self.neurons = neurons
        self.kappa = kappa
        self.n = n

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the current of the mean pulse into every neuron.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        theta = s.neurons.theta
        normalization = 2**self.n * math.factorial(self.n) ** 2 / math.factorial(2 * self.n)
        pulses = mean(normalization * (1 - cos(theta)) ** self.n)
        return [RateLaw(theta, (1 + cos(theta)) * s.kappa * pulses)]
