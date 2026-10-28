"""Neuroscience models: networks of spiking neurons and of relaxation oscillators."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, exp, oo, tanh
from sage.sets.real_set import RealSet
from scipy.ndimage import convolve

from ....core.elements import Element, FieldElement
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, CrossingLaw, RateLaw
from ....core.laws.utils import delayed, laplacian, maximum
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....engine.runtime import register_model
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....types import Array

NEAREST = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
LEGION_OBJECTS = ((1, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 1))
SILENT = -2.0


def _step(u: Expression, width: Expression) -> Expression:
    """Return a smooth step from 0 to 1 around `u = 0`, of a given width."""
    return 1 / (1 + exp(-u / width))


def _random_connections(n_neurons: int, probability: float, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Return the directed links of a random network without self-links."""
    return [
        (source, target)
        for source in range(n_neurons)
        for target in range(n_neurons)
        if source != target and rng.random() < probability
    ]


def _link_weights(stimulus: Array, total: float) -> Array:
    """Return the weight of each link into every site, a total weight shared among its stimulated neighbours."""
    neighbours = convolve(stimulus, NEAREST, mode='constant')
    return total / np.maximum(neighbours, 1.0)


class SpikingNeuronElement(Element):
    """Leaky integrate-and-fire neuron with an exponential synaptic trace of its own spikes.

    The potential follows `tau v' = -v + mu` plus synaptic input. When it reaches `threshold` it is reset and the
    trace `s` jumps by `1 / tau_s`, then decays as `s' = -s / tau_s`, so every spike carries a unit area of trace.
    Units are mV and ms.

    Args:
        label:
            The label of the neuron.

        v:
            The initial potential.

        s:
            The initial synaptic trace.

        tau:
            The membrane time constant.

        threshold:
            The firing threshold.

        reset:
            The potential after a spike.

        mu:
            The constant external drive, the potential the membrane relaxes to.

        tau_s:
            The decay time of the synaptic trace.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import SpikingNeuronElement
        >>> SpikingNeuronElement('neuron').threshold
        20.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[float, StateSpec()] = 0.0,
        s: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20.0,
        threshold: Annotated[float, ParameterSpec()] = 20.0,
        reset: Annotated[float, ParameterSpec()] = 10.0,
        mu: Annotated[float, ParameterSpec()] = 25.0,
        tau_s: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v
        self.s = s
        self.tau = tau
        self.threshold = threshold
        self.reset = reset
        self.mu = mu
        self.tau_s = tau_s

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the leaky membrane, the decay of the trace, and the spike that resets the potential.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            RateLaw(s.v, (s.mu - s.v) / s.tau),
            RateLaw(s.s, -s.s / s.tau_s),
            CrossingLaw(s.v - s.threshold, {s.v: s.reset, s.s: s.s + 1 / s.tau_s}, direction=1, label='spike'),
        ]


class DelayedSynapseInteraction(Interaction):
    """Synapse that adds the presynaptic trace from a transmission delay ago to the postsynaptic potential.

    The target gains `J s_source(t - D)`, so every presynaptic spike moves the postsynaptic potential by `J` in
    total, spread over the decay time of the trace and arriving after the delay `D`.

    Args:
        label:
            The label of the interaction.

        source:
            The path of the presynaptic neuron.

        target:
            The path of the postsynaptic neuron.

        J:
            The efficacy in mV, positive for excitation and negative for inhibition.

        D:
            The transmission delay in ms.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import DelayedSynapseInteraction
        >>> DelayedSynapseInteraction('synapse', source='a', target='b', J=-1.0).D
        1.5
    """

    def __init__(
        self,
        label: str,
        *,
        source: Annotated[str, RoleSpec()],
        target: Annotated[str, RoleSpec()],
        J: Annotated[float, ParameterSpec()] = 0.5,
        D: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.5,
    ) -> None:
        super().__init__(label)
        self.source = source
        self.target = target
        self.J = J
        self.D = D

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the delayed synaptic input `J s_source(t - D)`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.target.v, s.J * delayed(s.source.s, s.D))]


class TermanWangElement(Element):
    """Relaxation oscillator of Terman and Wang, coupled to its lattice neighbours through their active phases.

    The fast variable follows `x' = 3x - x³ + 2 - y + I + W m (sum_k m_k H(x_k - theta_x))`, summed over the four
    nearest sites, and the slow one `y' = epsilon (gamma (1 + tanh(x / beta)) - y)`. `H` is a smooth step of width
    `kappa`, and `m` marks the stimulated sites, so only stimulated neighbours excite each other. A positive input `I`
    makes the oscillator oscillate and a negative one makes it excitable.

    Args:
        label:
            The label of the oscillator or the lattice field.

        x:
            The initial fast variable.

        y:
            The initial slow variable.

        I:
            The external input.

        m:
            1 for a stimulated site and 0 otherwise.

        W:
            The weight of each excitatory link into the site.

        epsilon:
            The time scale of the slow variable.

        gamma:
            The height of the slow nullcline.

        beta:
            The width of the slow nullcline's step.

        theta_x:
            The threshold of the active phase.

        kappa:
            The width of the smooth steps.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import TermanWangElement
        >>> TermanWangElement('oscillator').gamma
        6.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = SILENT,
        y: Annotated[float, StateSpec()] = 0.0,
        I: Annotated[float, ParameterSpec()] = 0.2,
        m: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        W: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 6.0,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.02,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 6.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        theta_x: Annotated[float, ParameterSpec()] = -0.5,
        kappa: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.02,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.I = I
        self.m = m
        self.W = W
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta = beta
        self.theta_x = theta_x
        self.kappa = kappa

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the fast variable with its excitation by the neighbours, and the slow recovery.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        active = s.m * _step(s.x - s.theta_x, s.kappa)
        excitation = s.W * s.m * (laplacian(active) + 4 * active)
        return [
            RateLaw(s.x, 3 * s.x - s.x**3 + 2 - s.y + s.I + excitation),
            RateLaw(s.y, s.epsilon * (s.gamma * (1 + tanh(s.x / s.beta)) - s.y)),
        ]


class InhibitorElement(Element):
    """Global inhibitor of a LEGION network, whose activity `z` decays at rate `phi` without drive.

    Args:
        label:
            The label of the inhibitor.

        z:
            The initial activity.

        phi:
            The rate of the inhibitor.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import InhibitorElement
        >>> InhibitorElement('inhibitor').phi
        3.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        z: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        phi: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.z = z
        self.phi = phi

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the decay of the inhibitor.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.z, -s.phi * s.z)]


class GlobalInhibitionInteraction(Interaction):
    """Coupling of a lattice of oscillators to a global inhibitor that the most active oscillator drives.

    The inhibitor follows `z' = phi (sigma - z)`, where `sigma` is the largest activity `m H(x - theta_x)` of the
    oscillators, so it is driven when at least one stimulated oscillator is active. Once the inhibitor exceeds
    `theta_z` it inhibits every oscillator by `W_z`.

    Args:
        label:
            The label of the interaction.

        inhibitor:
            The path of the inhibitor.

        oscillators:
            The path of the lattice of oscillators.

        W_z:
            The strength of the inhibition.

        theta_z:
            The threshold of the inhibitor.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import GlobalInhibitionInteraction
        >>> GlobalInhibitionInteraction('inhibition', inhibitor='z', oscillators='grid').W_z
        1.5
    """

    def __init__(
        self,
        label: str,
        *,
        inhibitor: Annotated[str, RoleSpec()],
        oscillators: Annotated[str, RoleSpec()],
        W_z: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
        theta_z: Annotated[float, ParameterSpec()] = 0.1,
    ) -> None:
        super().__init__(label)
        self.inhibitor = inhibitor
        self.oscillators = oscillators
        self.W_z = W_z
        self.theta_z = theta_z

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the drive of the inhibitor by the most active oscillator and the inhibition of every oscillator.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        oscillator, inhibitor = s.oscillators, s.inhibitor
        activity = maximum(oscillator.m * _step(oscillator.x - oscillator.theta_x, oscillator.kappa))
        return [
            RateLaw(oscillator.x, -s.W_z * _step(inhibitor.z - s.theta_z, oscillator.kappa)),
            RateLaw(inhibitor.z, inhibitor.phi * activity),
        ]


@register_model(
    field='biology/neuroscience',
    tags=('neuron', 'spiking', 'network', 'delay', 'events'),
    references=('Brunel and Hakim 1999', 'arXiv:2112.12147'),
    defaults={'until': 100.0, 'dt': 0.5},
)
def delayed_lif_network(
    n_neurons: int = 10,
    *,
    probability: float = 0.3,
    J: float = 0.5,
    D: float = 1.5,
    mu: float = 25.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a random network of leaky integrate-and-fire neurons whose spikes reach their targets after a delay.

    Every neuron is driven above threshold and fires on its own with period `tau ln((mu - reset) / (mu -
    threshold))`. Each directed link exists with a probability and moves the target's potential by `J` per spike,
    after the delay `D`. Excitation speeds the firing up and inhibition slows it down.

    Args:
        n_neurons:
            The number of neurons.

        probability:
            The probability of each directed link.

        J:
            The efficacy of every link in mV.

        D:
            The transmission delay in ms.

        mu:
            The constant drive in mV.

        random_state:
            The seed or generator of the links and the initial potentials.

    Returns:
        system:
            The model, with the potentials `neuron_<i>.v`, the traces `neuron_<i>.s`, and the spikes as `spike`
            events.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import delayed_lif_network
        >>> len(delayed_lif_network(5).elements)
        5
    """
    rng = np.random.default_rng(random_state)
    potentials = rng.uniform(10.0, 20.0, n_neurons)
    neurons = [SpikingNeuronElement(f'neuron_{index}', v=float(potentials[index]), mu=mu) for index in range(n_neurons)]
    synapses = [
        DelayedSynapseInteraction(
            f'synapse_{source}_{target}',
            source=f'neuron_{source}',
            target=f'neuron_{target}',
            J=J,
            D=D,
        )
        for source, target in _random_connections(n_neurons, probability, rng)
    ]
    return System(elements=neurons, interactions=synapses)


@register_model(
    field='biology/neuroscience',
    tags=('oscillators', 'synchronization', 'segmentation', 'lattice', 'field'),
    references=('Terman and Wang 1995', 'Wang and Terman 1995', 'Gros 2015'),
    defaults={'until': 150.0, 'dt': 0.5},
)
def legion(
    stimulus: Sequence[Sequence[float]] = LEGION_OBJECTS,
    *,
    epsilon: float = 0.1,
    W: float = 6.0,
    W_z: float = 1.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a LEGION network, a lattice of relaxation oscillators that segments a stimulus into its objects.

    Stimulated oscillators excite their stimulated neighbours, so each connected object synchronizes, and a global
    inhibitor lets only one object be active at a time, so different objects take turns. The inhibitor is driven by the
    largest activity of the oscillators, so one active oscillator drives it fully. Unstimulated sites have a negative
    input and stay silent.

    The default `epsilon = 0.1` gives a cycle of about 60 time units, so with the default stimulus the second object
    first fires near `t = 51` and the first near `t = 70`. The value 0.02 of Terman and Wang makes the slow phases five
    times longer, and then no object starts an active phase before `t = 220`.

    Args:
        stimulus:
            The lattice of 1 for stimulated and 0 for unstimulated sites, by rows. The default holds two objects.

        epsilon:
            The time scale of the slow variables.

        W:
            The total excitatory weight into a stimulated site, shared among its stimulated neighbours.

        W_z:
            The strength of the global inhibition.

        random_state:
            The seed or generator of the initial states.

    Returns:
        system:
            The model, with the fast variables `oscillators.x`, the slow variables `oscillators.y`, and the inhibitor
            `inhibitor.z`.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import legion
        >>> legion().elements[0].space.shape
        (4, 4)
    """
    marks = np.asarray(stimulus, dtype=float)
    rng = np.random.default_rng(random_state)
    values = {
        'x': rng.uniform(-2.0, 2.0, marks.shape),
        'y': rng.uniform(0.0, 6.0, marks.shape),
        'I': np.where(marks > 0, 0.2, -0.02),
        'm': marks,
        'W': _link_weights(marks, W),
        'epsilon': epsilon,
    }
    oscillators = FieldElement(
        'oscillators',
        element=TermanWangElement,
        space=LatticeSpace(marks.shape),
        boundary={'x': ('fixed', SILENT), 'y': 'zero-flux'},
        values=values,
    )
    inhibition = GlobalInhibitionInteraction('inhibition', inhibitor='inhibitor', oscillators='oscillators', W_z=W_z)
    return System(elements=[oscillators, InhibitorElement('inhibitor')], interactions=[inhibition])
