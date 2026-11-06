"""Pulse coupling along plastic synapses whose efficacy follows the timing of the spikes at both ends."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import NeighborInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import where
from ....core.specs import ParameterSpec, RoleSpec


class SpikeTimingPlasticityInteraction(NeighborInteraction):
    """Pulse coupling along plastic synapses whose efficacy follows the timing of the spikes at both ends.

    The presynaptic neuron at the source of an edge drives the postsynaptic neuron at its target with the current
    `(1 + cos theta_post) kappa P(theta_pre) / k`, where `k` is the mean in-degree. The efficacy grows when the
    postsynaptic neuron fires after the presynaptic one and shrinks when it fires before, `dkappa/dt = epsilon (A_plus
    a_pre P(theta_post) - A_minus b_post P(theta_pre))`, the window `W(dt) = A_plus exp(-dt / tau_plus)` for `dt > 0`
    and `-A_minus exp(dt / tau_minus)` for `dt < 0` applied to every pair of spikes through the traces. Soft bounds
    scale potentiation by `kappa_max - kappa` and depression by `kappa`, so the efficacy stays between 0 and
    `kappa_max`.

    Args:
        label:
            The label of the interaction.

        neurons:
            The path of the group of traced theta neurons on the nodes.

        synapses:
            The path of the group of synapses on the edges.

        k:
            The mean in-degree that normalizes the coupling.

        epsilon:
            The learning rate.

        A_plus:
            The amplitude of potentiation.

        A_minus:
            The amplitude of depression.

        kappa_max:
            The largest efficacy.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import SpikeTimingPlasticityInteraction
        >>> SpikeTimingPlasticityInteraction('stdp', neurons='neurons', synapses='synapses').A_minus
        1.05
    """

    def __init__(
        self,
        label: str,
        *,
        neurons: Annotated[str, RoleSpec()],
        synapses: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        A_plus: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        A_minus: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.05,
        kappa_max: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.neurons = neurons
        self.synapses = synapses
        self.k = k
        self.epsilon = epsilon
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.kappa_max = kappa_max

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the pulse current into the postsynaptic neuron and the plasticity of the efficacy.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        pre, post, kappa = s.source, s.target, s.edge.kappa
        pre_pulse = 2 * (1 - cos(pre.theta)) ** 2 / 3
        post_pulse = 2 * (1 - cos(post.theta)) ** 2 / 3
        change = s.epsilon * (s.A_plus * pre.a * post_pulse - s.A_minus * post.b * pre_pulse)
        bounded = where(change > 0, change * (s.kappa_max - kappa), change * kappa)
        return [
            RateLaw(post.theta, (1 + cos(post.theta)) * kappa * pre_pulse / s.k),
            RateLaw(kappa, bounded / s.kappa_max),
        ]
