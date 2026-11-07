"""Neuroscience systems: networks whose synapses adapt to the activity of the neurons they join."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from itertools import permutations
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import cos, oo, pi, sin, tanh
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement, LatticeElement
from ....core.components.interactions import Interaction, NeighborInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import clip, delayed, where
from ....core.components.spaces import NetworkSpace
from ....core.components.system import System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ._theta_neuron import ThetaNeuronElement


class AxonElement(Element):
    """Axon on an edge of a network, whose conduction delay myelination changes.

    Args:
        label:
            The label of the axon or of the group on the edges.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        tau:
            The initial conduction delay.

        tau0:
            The baseline delay that the delay relaxes to.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import AxonElement
        >>> AxonElement('axon', tau0=2.0).tau0
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        tau: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        tau0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.tau = tau
        self.tau0 = tau0


class PhaseUnitElement(Element):
    """Neural unit reduced to a phase that turns at its natural frequency, `theta' = omega`.

    Couplings to other units add to the rate of the phase.

    Args:
        label:
            The label of the unit or of the group of units.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        theta:
            The initial phase.

        omega:
            The natural frequency.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import PhaseUnitElement
        >>> PhaseUnitElement('unit', omega=1.5).omega
        1.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.omega = omega

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rotation of the phase.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.theta, s.omega)]


class SynapseElement(Element):
    """Synapse on an edge of a network, with an efficacy that plasticity changes.

    Args:
        label:
            The label of the synapse or of the group on the edges.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        kappa:
            The initial efficacy.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import SynapseElement
        >>> SynapseElement('synapse', kappa=0.5).kappa
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        kappa: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.kappa = kappa


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


class DelayAdaptationInteraction(Interaction):
    """Phase coupling through conduction delays that adapt to the phase differences of the units they join.

    The target unit reads the phase of the source unit along the axon as it was one conduction delay ago,
    `dtheta_i/dt += kappa sin(theta_j(t - tau_ij) - theta_i)`. The delay relaxes to its baseline and is pushed by the
    phase difference, `dtau_ij/dt = epsilon H(tau_ij) (-(tau_ij - tau0_ij) + rho sin(theta_j - theta_i))`, where the
    smooth step `H(tau) = (1 + tanh(tau / width)) / 2` keeps the delay nonnegative. The coupling reads the delay no
    shorter than `shortest`, which bounds the steps of the solver.

    Args:
        label:
            The label of the interaction.

        source:
            The path of the unit that sends the signal.

        target:
            The path of the unit that receives the signal.

        axon:
            The path of the axon that conducts it.

        kappa:
            The coupling strength of an edge.

        epsilon:
            The adaptation rate of the delays.

        rho:
            The strength of the push of the phase difference on the delay.

        width:
            The width of the smooth step.

        shortest:
            The shortest delay the coupling reads.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import DelayAdaptationInteraction
        >>> DelayAdaptationInteraction('myelination', source='a', target='b', axon='ab', rho=0.5).rho
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        source: Annotated[str, RoleSpec()],
        target: Annotated[str, RoleSpec()],
        axon: Annotated[str, RoleSpec()],
        kappa: Annotated[float, ParameterSpec()] = 0.1,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        rho: Annotated[float, ParameterSpec()] = 1.0,
        width: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.05,
        shortest: float = 0.1,
    ) -> None:
        super().__init__(label)
        self.source = source
        self.target = target
        self.axon = axon
        self.kappa = kappa
        self.epsilon = epsilon
        self.rho = rho
        self.width = width
        self.shortest = shortest

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the delayed coupling of the target and the adaptation of the delay.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        source, target, tau = s.source.theta, s.target.theta, s.axon.tau
        step = (1 + tanh(tau / s.width)) / 2
        return [
            RateLaw(target, s.kappa * sin(delayed(source, clip(tau, low=self.shortest), self.shortest) - target)),
            RateLaw(tau, s.epsilon * step * (-(tau - s.axon.tau0) + s.rho * sin(source - target))),
        ]


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


@register_system(
    field='biology/neuroscience',
    tags=('plasticity', 'stdp', 'adaptive network', 'theta neurons', 'ode'),
    references=('Song, Miller, and Abbott 2000', 'Berner et al. 2023'),
    defaults={'until': 150.0, 'dt': 0.5},
)
def stdp_network(
    n_neurons: int = 20,
    *,
    p: float = 0.3,
    eta: float = 0.5,
    spread: float = 0.2,
    epsilon: float = 0.05,
    A_plus: float = 1.0,
    A_minus: float = 1.05,
    tau: float = 2.0,
    kappa: float = 0.5,
    seed: int | None = 0,
) -> System:
    """Return theta neurons on a random directed network whose synapses learn by spike-timing-dependent plasticity.

    The neurons fire periodically with excitabilities spread around `eta`, and every synapse is potentiated when its
    presynaptic neuron fires shortly before its postsynaptic neuron and depressed when it fires shortly after. Spikes
    of neurons that fire at different rates fall at all delays, so the mean change of the efficacies follows the
    integral of the window, `(A_plus - A_minus) tau`: a window dominated by depression weakens the synapses and one
    dominated by potentiation strengthens them, while the spread of the efficacies grows.

    Args:
        n_neurons:
            The number of neurons.

        p:
            The probability of every directed synapse.

        eta:
            The mean excitability.

        spread:
            The half-width of the uniform spread of the excitabilities.

        epsilon:
            The learning rate.

        A_plus:
            The amplitude of potentiation.

        A_minus:
            The amplitude of depression.

        tau:
            The time of both windows of the plasticity.

        kappa:
            The initial efficacy of every synapse.

        seed:
            The seed of the network, the excitabilities, and the initial phases.

    Returns:
        system:
            The system, with the phases `neurons.theta`, the traces `neurons.a` and `neurons.b`, and the efficacies
            `synapses.kappa` on the edges.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import stdp_network
        >>> stdp_network(10).elements[0].prepare().size_
        10
    """
    rng = np.random.default_rng(seed)
    graph = nx.gnp_random_graph(n_neurons, p, seed=int(rng.integers(2**31)), directed=True)
    space = NetworkSpace(graph)
    neurons = LatticeElement(
        'neurons',
        element=TracedThetaNeuronElement,
        space=space,
        values={
            'theta': rng.uniform(0, 2 * np.pi, n_neurons),
            'eta': eta + rng.uniform(-spread, spread, n_neurons),
            'tau_plus': tau,
            'tau_minus': tau,
        },
    )
    synapses = GroupElement('synapses', element=SynapseElement, space=space, on='edges', values={'kappa': kappa})
    plasticity = SpikeTimingPlasticityInteraction(
        'stdp',
        neurons='neurons',
        synapses='synapses',
        k=max(p * (n_neurons - 1), 1.0),
        epsilon=epsilon,
        A_plus=A_plus,
        A_minus=A_minus,
    )
    return System(elements=[neurons, synapses], interactions=[plasticity])


@register_system(
    field='biology/neuroscience',
    tags=('adaptive delays', 'myelination', 'phase oscillators', 'adaptive network', 'dde'),
    references=('Park and Lefebvre 2020', 'Berner et al. 2023'),
    defaults={'until': 300.0, 'dt': 0.5},
)
def adaptive_delay_network(
    n_units: int = 5,
    *,
    omega: float = 1.0,
    spread: float = 0.1,
    kappa: float = 1.0,
    epsilon: float = 0.1,
    rho: float = 1.0,
    tau0: float = 3.0,
    seed: int | None = 0,
) -> System:
    """Return phase units coupled all to all through conduction delays that adapt to their activity, as myelination.

    Unit `i` follows `dtheta_i/dt = omega_i + (kappa / N) sum_j sin(theta_j(t - tau_ij) - theta_i)`, and every delay
    `dtau_ij/dt = epsilon H(tau_ij) (-(tau_ij - tau0) + rho sin(theta_j - theta_i))` relaxes to its baseline `tau0`
    and is pushed by the phase difference of its units. With the defaults, fixed delays of 3 leave the units
    incoherent, and the adaptation of the delays restores a partial synchrony.

    Args:
        n_units:
            The number of units.

        omega:
            The mean natural frequency.

        spread:
            The standard deviation of the natural frequencies.

        kappa:
            The coupling strength, divided by the number of units.

        epsilon:
            The adaptation rate of the delays.

        rho:
            The strength of the push of the phase difference on the delays.

        tau0:
            The baseline delay of every edge.

        seed:
            The seed of the frequencies and the initial phases.

    Returns:
        system:
            The system, with the phases `unit_<i>.theta` of the units and the delays `axon_<i>_<j>.tau` of the axons
            from unit `i` to unit `j`.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import adaptive_delay_network
        >>> len(adaptive_delay_network(4).interactions)
        12
    """
    rng = np.random.default_rng(seed)
    phases, frequencies = rng.uniform(0, 2 * np.pi, n_units), rng.normal(omega, spread, n_units)
    units = [
        PhaseUnitElement(f'unit_{index}', theta=float(phases[index]), omega=float(frequencies[index]))
        for index in range(n_units)
    ]
    pairs = list(permutations(range(n_units), 2))
    axons = [AxonElement(f'axon_{source}_{target}', tau=tau0, tau0=tau0) for source, target in pairs]
    adaptations = [
        DelayAdaptationInteraction(
            f'myelination_{source}_{target}',
            source=f'unit_{source}',
            target=f'unit_{target}',
            axon=f'axon_{source}_{target}',
            kappa=kappa / n_units,
            epsilon=epsilon,
            rho=rho,
        )
        for source, target in pairs
    ]
    return System(elements=[*units, *axons], interactions=adaptations)
