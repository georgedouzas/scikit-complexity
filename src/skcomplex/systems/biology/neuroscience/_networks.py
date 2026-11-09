"""Neuroscience systems: networks of spiking neurons and of relaxation oscillators."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import Expression, oo, tanh
from sage.sets.real_set import RealSet
from scipy.ndimage import convolve

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import CouplingInteraction, RateInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import delayed, logistic
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, System
from ....core.components.types import Target
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..common.excitable import SpikeResetInteraction
from ._leaky_integration import LeakyIntegrationInteraction

SILENT = -2.0

NEAREST = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

LEGION_OBJECTS = ((1, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 1))

THRESHOLD = 20.0

RESET = 10.0

TRACE_TIME = 2.0


class InhibitorElement(Element):
    """Global inhibitor of a LEGION network.

    Its activity `z` decays at rate `phi` without drive.

    Args:
        label:
            The label of the inhibitor.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        z:
            The initial activity.

        phi:
            The rate of the inhibitor.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import InhibitorElement
        >>> InhibitorElement('inhibitor').phi
        3.0
    """

    law_classes = (RateLaw,)

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


class SpikingNeuronElement(Element):
    """Leaky integrate-and-fire neuron with a synaptic trace of its own spikes.

    The trace is exponential.

    The potential follows `tau v' = -v + mu` plus synaptic input, and the trace decays as `s' = -s / tau_s`. A spike
    reset sets the potential back and raises the trace by `1 / tau_s`, so every spike carries a unit area of trace.
    Units are mV and ms. The element holds the states and parameters and has no laws of its own: the leak and trace
    interactions of `DelayedLIFNetworkSystem` read its parameters.

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


class TermanWangElement(Element):
    """Relaxation oscillator of Terman and Wang.

    It is the dynamics of one site of a LEGION lattice.

    The fast variable follows `x' = 3x - x³ + 2 - y + I` and the slow one `y' = epsilon (gamma (1 + tanh(x / beta)) -
    y)`. A positive input `I` makes the oscillator oscillate and a negative one makes it excitable. The parameters `m`,
    `W`, `theta_x`, and `kappa` belong to the coupling of a lattice of oscillators, a `CouplingInteraction` that adds
    `W m (sum_k m_k H(x_k - theta_x))` over the four nearest sites, where `H` is a smooth step of width `kappa` and `m`
    marks the stimulated sites, so only stimulated neighbours excite each other.

    Args:
        label:
            The label of the oscillator or the lattice field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.systems.biology.neuroscience import TermanWangElement
        >>> TermanWangElement('oscillator').gamma
        6.0
    """

    law_classes = (RateLaw,)

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
        """Return the fast variable and the slow recovery.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            RateLaw(s.x, 3 * s.x - s.x**3 + 2 - s.y + s.I),
            RateLaw(s.y, s.epsilon * (s.gamma * (1 + tanh(s.x / s.beta)) - s.y)),
        ]


class DelayedSynapseInteraction(RateInteraction):
    """Synapse that adds the delayed presynaptic trace to the postsynaptic potential.

    The trace is the one from a transmission delay ago.

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
        >>> from skcomplex.systems.biology.neuroscience import DelayedSynapseInteraction
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
        super().__init__(label, target=target, state='v')
        self.source = source
        self.J = J
        self.D = D

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the delayed synaptic input `J s_source(t - D)`.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the namespaces of the source and the target.

        Returns:
            rate:
                The rate of the postsynaptic potential.
        """
        return namespace.J * delayed(namespace.source.s, namespace.D)


class GlobalInhibitionInteraction(CouplingInteraction):
    """Inhibition of every oscillator of a lattice by a global inhibitor.

    The inhibition acts once the inhibitor exceeds its threshold.

    Every oscillator gets `-W_z H(z - theta_z)` from the inhibitor, a smooth step of width `kappa` of the oscillator.
    It is the coupling of the members of a group to a hub element, and `InhibitorDriveInteraction` drives the hub.

    Args:
        label:
            The label of the interaction.

        oscillators:
            The path of the lattice of oscillators.

        inhibitor:
            The path of the inhibitor.

        W_z:
            The strength of the inhibition.

        theta_z:
            The threshold of the inhibitor.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import GlobalInhibitionInteraction
        >>> GlobalInhibitionInteraction('inhibition', inhibitor='z', oscillators='grid').W_z
        1.5
    """

    def __init__(
        self,
        label: str,
        *,
        oscillators: Annotated[str, RoleSpec()],
        inhibitor: Annotated[str, RoleSpec()],
        W_z: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
        theta_z: Annotated[float, ParameterSpec()] = 0.1,
    ) -> None:
        super().__init__(
            label,
            target=oscillators,
            source=inhibitor,
            write='x',
            function=self._get_inhibition,
            mode='group',
        )
        self.oscillators = oscillators
        self.inhibitor = inhibitor
        self.W_z = W_z
        self.theta_z = theta_z

    @staticmethod
    def _get_inhibition(namespace: SimpleNamespace) -> Target:
        """Return the inhibition `-W_z H(z - theta_z)` of an oscillator."""
        return -namespace.W_z * logistic(namespace.other.z - namespace.theta_z, 0, namespace.member.kappa)


class InhibitorDriveInteraction(CouplingInteraction):
    """Drive of a global inhibitor by the largest activity of the lattice oscillators.

    The activity is `m H(x - theta_x)`.

    The inhibitor gets `phi max_j m_j H(x_j - theta_x)`, with the rate `phi` of the inhibitor and a smooth step of
    width `kappa` of every oscillator, so it is driven when at least one stimulated oscillator is active. It is the
    coupling of a hub element to a group by the largest value over the group.

    Args:
        label:
            The label of the interaction.

        inhibitor:
            The path of the inhibitor.

        oscillators:
            The path of the lattice of oscillators.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import InhibitorDriveInteraction
        >>> InhibitorDriveInteraction('drive', inhibitor='z', oscillators='grid').reduce
        'max'
    """

    def __init__(
        self,
        label: str,
        *,
        inhibitor: Annotated[str, RoleSpec()],
        oscillators: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(
            label,
            target=inhibitor,
            source=oscillators,
            write='z',
            function=self._get_drive,
            reduce='max',
            mode='group',
        )
        self.inhibitor = inhibitor
        self.oscillators = oscillators

    @staticmethod
    def _get_drive(namespace: SimpleNamespace) -> Target:
        """Return the drive `phi m H(x - theta_x)` of the inhibitor by one oscillator."""
        other = namespace.other
        return namespace.member.phi * other.m * logistic(other.x - other.theta_x, 0, other.kappa)


class DelayedLIFNetworkSystem(BaseSystem):
    """A random network of leaky integrate-and-fire neurons with delayed spikes.

    The spikes reach their targets after a delay.

    Every neuron is driven above threshold and fires on its own with period `tau ln((mu - reset) / (mu - threshold))`.
    Each directed link exists with a probability and moves the target's potential by `J` per spike, after the delay `D`.
    Excitation speeds the firing up and inhibition slows it down. The system has the potentials `neuron_<i>.v`, the
    traces `neuron_<i>.s`, and the spikes of neuron `i` as `spike_<i>` events.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import DelayedLIFNetworkSystem
        >>> len(DelayedLIFNetworkSystem(5).prepare().system_.elements)
        5
    """

    field = 'biology/neuroscience'
    topics = ('neuron', 'spiking', 'network', 'delay', 'events')
    references = ('Brunel and Hakim 1999', 'arXiv:2112.12147')
    until = 100.0
    dt = 0.5

    def __init__(
        self,
        n_neurons: int = 10,
        *,
        probability: float = 0.3,
        J: float = 0.5,
        D: float = 1.5,
        mu: float = 25.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_neurons = n_neurons
        self.probability = probability
        self.J = J
        self.D = D
        self.mu = mu
        self.random_state = random_state

    @staticmethod
    def _random_connections(n_neurons: int, probability: float, rng: np.random.Generator) -> list[tuple[int, int]]:
        """Return the directed links of a random network without self-links."""
        return list(nx.gnp_random_graph(n_neurons, probability, seed=rng, directed=True).edges)

    def _build_system(self) -> System:
        """Return the parts of a random network of leaky integrate-and-fire neurons."""
        rng = np.random.default_rng(self.random_state)
        potentials = rng.uniform(10.0, 20.0, self.n_neurons)
        neurons = [
            SpikingNeuronElement(f'neuron_{index}', v=float(potentials[index]), mu=self.mu, tau_s=TRACE_TIME)
            for index in range(self.n_neurons)
        ]
        leaks = [
            LeakyIntegrationInteraction(
                f'leak_{index}',
                target=neuron.label,
                state='v',
                goal=lambda s: s.target.mu,
                time=lambda s: s.target.tau,
            )
            for index, neuron in enumerate(neurons)
        ]
        traces = [
            RateInteraction(
                f'trace_decay_{index}',
                target=neuron.label,
                state='s',
                rate=lambda s: -s.target.s / s.target.tau_s,
            )
            for index, neuron in enumerate(neurons)
        ]
        spikes = [
            SpikeResetInteraction(
                f'spike_{index}',
                target=neuron.label,
                threshold=THRESHOLD,
                after=RESET,
                kicked='s',
                kick=1 / TRACE_TIME,
            )
            for index, neuron in enumerate(neurons)
        ]
        synapses = [
            DelayedSynapseInteraction(
                f'synapse_{source}_{target}',
                source=f'neuron_{source}',
                target=f'neuron_{target}',
                J=self.J,
                D=self.D,
            )
            for source, target in self._random_connections(self.n_neurons, self.probability, rng)
        ]
        return System(self.label, elements=neurons, interactions=[*leaks, *traces, *spikes, *synapses])


class LegionSystem(BaseSystem):
    """A LEGION network, a lattice of relaxation oscillators.

    The network segments a stimulus into its objects.

    Stimulated oscillators excite their stimulated neighbours, so each connected object synchronizes, and a global
    inhibitor lets only one object be active at a time, so different objects take turns. The inhibitor is driven by the
    largest activity of the oscillators, so one active oscillator drives it fully. Unstimulated sites have a negative
    input and stay silent.

    The default `epsilon = 0.1` gives a cycle of about 60 time units, so with the default stimulus the second object
    first fires near `t = 51` and the first near `t = 70`. The value 0.02 of Terman and Wang makes the slow phases five
    times longer, and then no object starts an active phase before `t = 220`. The system has the fast variables
    `oscillators.x`, the slow variables `oscillators.y`, and the inhibitor `inhibitor.z`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import LegionSystem
        >>> LegionSystem().prepare().system_.elements[0].space.shape
        (4, 4)
    """

    field = 'biology/neuroscience'
    topics = ('oscillators', 'synchronization', 'segmentation', 'lattice', 'field')
    references = ('Terman and Wang 1995', 'Wang and Terman 1995', 'Gros 2015')
    until = 150.0
    dt = 0.5

    def __init__(
        self,
        stimulus: Sequence[Sequence[float]] = LEGION_OBJECTS,
        *,
        epsilon: float = 0.1,
        W: float = 6.0,
        W_z: float = 1.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.stimulus = stimulus
        self.epsilon = epsilon
        self.W = W
        self.W_z = W_z
        self.random_state = random_state

    @staticmethod
    def _excitation(s: SimpleNamespace) -> Expression:
        """Return the excitation of an oscillator by a stimulated neighbour.

        The neighbour excites it in its active phase.
        """
        active = s.other.m * logistic(s.other.x - s.other.theta_x, 0, s.other.kappa)
        excitation: Expression = s.member.W * s.member.m * active
        return excitation

    @staticmethod
    def _link_weights(stimulus: Array, total: float) -> Array:
        """Return the weight of each link into every site.

        A total weight is shared among the stimulated neighbours of the site.
        """
        neighbours = convolve(stimulus, NEAREST, mode='constant')
        return total / np.maximum(neighbours, 1.0)

    def _build_system(self) -> System:
        """Return the parts of a LEGION network."""
        marks = np.asarray(self.stimulus, dtype=float)
        rng = np.random.default_rng(self.random_state)
        values = {
            'x': rng.uniform(-2.0, 2.0, marks.shape),
            'y': rng.uniform(0.0, 6.0, marks.shape),
            'I': np.where(marks > 0, 0.2, -0.02),
            'm': marks,
            'W': self._link_weights(marks, self.W),
            'epsilon': self.epsilon,
        }
        oscillators = FieldElement(
            'oscillators',
            element=TermanWangElement,
            space=LatticeSpace(marks.shape),
            boundary={'x': ('fixed', SILENT), 'y': 'zero-flux'},
            values=values,
        )
        excitation = CouplingInteraction(
            'excitation',
            target='oscillators',
            read='x',
            write='x',
            function=self._excitation,
        )
        drive = InhibitorDriveInteraction('drive', inhibitor='inhibitor', oscillators='oscillators')
        inhibition = GlobalInhibitionInteraction(
            'inhibition',
            inhibitor='inhibitor',
            oscillators='oscillators',
            W_z=self.W_z,
        )
        return System(
            self.label,
            elements=[oscillators, InhibitorElement('inhibitor')],
            interactions=[excitation, drive, inhibition],
        )
