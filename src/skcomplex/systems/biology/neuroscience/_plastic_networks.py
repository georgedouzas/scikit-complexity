"""Neuroscience systems: networks with adaptive synapses.

The synapses adapt to the activity of the neurons they join.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from itertools import permutations
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import cos, oo, sin, tanh
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, Period
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import clip, where
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ..common.excitable import ThetaNeuronElement, ThetaNeuronInteraction


class AxonElement(Element):
    """Axon on an edge of a network, whose conduction delay myelination changes.

    Args:
        label:
            The label of the axon or of the group on the edges.

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
        *,
        tau: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        tau0: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.tau = tau
        self.tau0 = tau0


class PhaseUnitElement(Element):
    """Neural unit reduced to a phase `theta` that wraps around, with its natural frequency `omega`.

    Args:
        label:
            The label of the unit or of the group of units.

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
        *,
        theta: Annotated[float, Period(2 * math.pi)] = 0.0,
        omega: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.theta = theta
        self.omega = omega


class RotationInteraction(Interaction):
    """Rotation of the phase of a unit at its natural frequency, `theta' = omega`.

    Couplings to other units add to the rate of the phase.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the unit, whose values are `theta` and `omega`.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import RotationInteraction
        >>> RotationInteraction('rotation', target='unit_0').target
        'unit_0'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rotation of the phase of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        self.laws_ = [RateLaw(context.target.theta, context.target.omega)]


class SynapseElement(Element):
    """Synapse on an edge of a network, with an efficacy that plasticity changes.

    Args:
        label:
            The label of the synapse or of the group on the edges.

        kappa:
            The initial efficacy.

        space:
            The network whose edges hold one synapse each, or `None` for one synapse.

        on:
            `'edges'` for one synapse per edge of the network, or `None` for one synapse.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import SynapseElement
        >>> SynapseElement('synapse', kappa=0.5).kappa
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        kappa: Annotated[float, RealSet.closed_open(0, oo)] = 0.5,
        space: BaseSpace | None = None,
        on: str | None = None,
    ) -> None:
        super().__init__(label, space=space, on=on)
        self.kappa = kappa


class TracedThetaNeuronElement(ThetaNeuronElement):
    """Theta neurons with a presynaptic trace `a` and a postsynaptic trace `b` of their own spikes.

    Args:
        label:
            The label of the neuron or the group.

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

        space:
            The network whose nodes hold one neuron each, or `None` for one neuron.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import TracedThetaNeuronElement
        >>> TracedThetaNeuronElement('neuron').tau_plus
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        theta: Annotated[float | Array, Period(2 * math.pi)] = 0.0,
        a: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.0,
        b: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.0,
        eta: float | Array = 1.0,
        tau_plus: Annotated[float, RealSet.open(0, oo)] = 2.0,
        tau_minus: Annotated[float, RealSet.open(0, oo)] = 2.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, theta=theta, eta=eta, space=space)
        self.a = a
        self.b = b
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus


class SpikeTraceInteraction(Interaction):
    """Traces of the spikes of theta neurons, which spike-timing-dependent plasticity reads.

    A spike is the smooth pulse `P(theta) = (1 - cos theta)^2 * 2 / 3`, of unit mean over the phase, which peaks as the
    phase passes π. The presynaptic trace `a` and the postsynaptic trace `b` rise with every pulse and decay with the
    times `tau_plus` and `tau_minus`, `da/dt = P(theta) - a / tau_plus` and `db/dt = P(theta) - b / tau_minus`, so
    they hold the recent spikes weighted by the exponential windows of the plasticity.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the neurons, whose values are `theta`, `a`, `b`, `tau_plus`, and `tau_minus`.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import SpikeTraceInteraction
        >>> SpikeTraceInteraction('traces', target='neurons').target
        'neurons'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the traces of the spikes of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        s = context.target
        pulse = 2 * (1 - cos(s.theta)) ** 2 / 3
        self.laws_ = [
            RateLaw(s.a, pulse - s.a / s.tau_plus),
            RateLaw(s.b, pulse - s.b / s.tau_minus),
        ]


class DelayAdaptationInteraction(Interaction):
    """Phase coupling through adaptive conduction delays.

    The delays adapt to the phase differences of the units they join.

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

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        source: Role,
        target: Role,
        axon: Role,
        kappa: float = 0.1,
        epsilon: Annotated[float, RealSet.closed_open(0, oo)] = 0.01,
        rho: float = 1.0,
        width: Annotated[float, RealSet.open(0, oo)] = 0.05,
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

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the delayed coupling of the target and the adaptation of the delay.

        Args:
            context:
                The namespace of the units and the axon.
        """
        super()._prepare(context)
        s, t = context, context.t
        source, target, tau = s.source.theta, s.target.theta, s.axon.tau
        step = (1 + tanh(tau / s.width)) / 2
        self.laws_ = [
            RateLaw(target.diff(t), s.kappa * sin(source(t=t - clip(tau, low=self.shortest)) - target)),
            RateLaw(tau.diff(t), s.epsilon * step * (-(tau - s.axon.tau0) + s.rho * sin(source - target))),
        ]


class SpikeTimingPlasticityInteraction(Interaction):
    """Pulse coupling along plastic synapses.

    The efficacy of a synapse follows the timing of the spikes at both ends.

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

        among:
            The path of the group of traced theta neurons on the nodes. The efficacies `kappa` are values of the group
            of synapses on the edges.

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

        along:
            The edges the laws run on, `True` for the edges of the network the neurons live on.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import SpikeTimingPlasticityInteraction
        >>> SpikeTimingPlasticityInteraction('stdp', among='neurons').A_minus
        1.05
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        among: Role,
        k: Annotated[float, RealSet.open(0, oo)] = 1.0,
        epsilon: Annotated[float, RealSet.closed_open(0, oo)] = 0.01,
        A_plus: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        A_minus: Annotated[float, RealSet.closed_open(0, oo)] = 1.05,
        kappa_max: Annotated[float, RealSet.open(0, oo)] = 1.0,
        along: bool | str = True,
    ) -> None:
        super().__init__(label, among=among, along=along)
        self.k = k
        self.epsilon = epsilon
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.kappa_max = kappa_max

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the pulse current into the postsynaptic neuron and the plasticity of the efficacy.

        Args:
            context:
                The namespace of a synapse, with its ends `context.source` and `context.target` and its member
                `context.edge` of the group on the edges.
        """
        super()._prepare(context)
        s = context
        pre, post, kappa = s.source, s.target, s.edge.kappa
        pre_pulse = 2 * (1 - cos(pre.theta)) ** 2 / 3
        post_pulse = 2 * (1 - cos(post.theta)) ** 2 / 3
        change = s.epsilon * (s.A_plus * pre.a * post_pulse - s.A_minus * post.b * pre_pulse)
        bounded = where(change > 0, change * (s.kappa_max - kappa), change * kappa)
        self.laws_: list[BaseLaw] = [
            RateLaw(post.theta, (1 + cos(post.theta)) * kappa * pre_pulse / s.k),
            RateLaw(kappa, bounded / s.kappa_max),
        ]


class STDPNetworkSystem(BaseSystem):
    """Theta neurons on a random directed network with plastic synapses.

    The synapses learn by spike-timing-dependent plasticity.

    The neurons fire periodically with excitabilities spread around `eta`, and every synapse is potentiated when its
    presynaptic neuron fires shortly before its postsynaptic neuron and depressed when it fires shortly after. Spikes of
    neurons that fire at different rates fall at all delays, so the mean change of the efficacies follows the integral
    of the window, `(A_plus - A_minus) tau`: a window dominated by depression weakens the synapses and one dominated by
    potentiation strengthens them, while the spread of the efficacies grows. The system has the phases `neurons.theta`,
    the traces `neurons.a` and `neurons.b`, and the efficacies `synapses.kappa` on the edges.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import STDPNetworkSystem
        >>> STDPNetworkSystem(10).prepare().system_.elements[0].theta.shape
        (10,)
    """

    field = 'biology/neuroscience'
    topics = ('plasticity', 'stdp', 'adaptive network', 'theta neurons', 'ode')
    references = ('Song, Miller, and Abbott 2000', 'Berner et al. 2023')
    until = 150.0
    dt = 0.5

    def __init__(
        self,
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
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_neurons = n_neurons
        self.p = p
        self.eta = eta
        self.spread = spread
        self.epsilon = epsilon
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau = tau
        self.kappa = kappa
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of theta neurons on a random directed network.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.seed)
        graph = nx.gnp_random_graph(self.n_neurons, self.p, seed=int(rng.integers(2**31)), directed=True)
        space = NetworkSpace(graph)
        neurons = TracedThetaNeuronElement(
            'neurons',
            theta=rng.uniform(0, 2 * np.pi, self.n_neurons),
            eta=self.eta + rng.uniform(-self.spread, self.spread, self.n_neurons),
            tau_plus=self.tau,
            tau_minus=self.tau,
            space=space,
        )
        synapses = SynapseElement('synapses', kappa=self.kappa, space=space, on='edges')
        plasticity = SpikeTimingPlasticityInteraction(
            'stdp',
            among='neurons',
            k=max(self.p * (self.n_neurons - 1), 1.0),
            epsilon=self.epsilon,
            A_plus=self.A_plus,
            A_minus=self.A_minus,
        )
        phases = ThetaNeuronInteraction('phases', target='neurons')
        traces = SpikeTraceInteraction('traces', target='neurons')
        self.system_ = System(
            self.label,
            elements=[neurons, synapses],
            interactions=[phases, traces, plasticity],
        )


class AdaptiveDelayNetworkSystem(BaseSystem):
    """Phase units coupled all to all through adaptive conduction delays.

    The delays adapt to the activity of the units, as myelination.

    Unit `i` follows `dtheta_i/dt = omega_i + (kappa / N) sum_j sin(theta_j(t - tau_ij) - theta_i)`, and every delay
    `dtau_ij/dt = epsilon H(tau_ij) (-(tau_ij - tau0) + rho sin(theta_j - theta_i))` relaxes to its baseline `tau0` and
    is pushed by the phase difference of its units. With the defaults, fixed delays of 3 leave the units incoherent, and
    the adaptation of the delays restores a partial synchrony. The system has the phases `unit_<i>.theta` of the units
    and the delays `axon_<i>_<j>.tau` of the axons from unit `i` to unit `j`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import AdaptiveDelayNetworkSystem
        >>> len(AdaptiveDelayNetworkSystem(4).prepare().system_.interactions)
        16
    """

    field = 'biology/neuroscience'
    topics = ('adaptive delays', 'myelination', 'phase oscillators', 'adaptive network', 'dde')
    references = ('Park and Lefebvre 2020', 'Berner et al. 2023')
    until = 300.0
    dt = 0.5

    def __init__(
        self,
        n_units: int = 5,
        *,
        omega: float = 1.0,
        spread: float = 0.1,
        kappa: float = 1.0,
        epsilon: float = 0.1,
        rho: float = 1.0,
        tau0: float = 3.0,
        seed: int | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_units = n_units
        self.omega = omega
        self.spread = spread
        self.kappa = kappa
        self.epsilon = epsilon
        self.rho = rho
        self.tau0 = tau0
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of phase units coupled all to all through adaptive delays.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.seed)
        phases, frequencies = rng.uniform(0, 2 * np.pi, self.n_units), rng.normal(self.omega, self.spread, self.n_units)
        units = [
            PhaseUnitElement(f'unit_{index}', theta=float(phases[index]), omega=float(frequencies[index]))
            for index in range(self.n_units)
        ]
        pairs = list(permutations(range(self.n_units), 2))
        axons = [AxonElement(f'axon_{source}_{target}', tau=self.tau0, tau0=self.tau0) for source, target in pairs]
        adaptations = [
            DelayAdaptationInteraction(
                f'myelination_{source}_{target}',
                source=f'unit_{source}',
                target=f'unit_{target}',
                axon=f'axon_{source}_{target}',
                kappa=self.kappa / self.n_units,
                epsilon=self.epsilon,
                rho=self.rho,
            )
            for source, target in pairs
        ]
        rotations = [RotationInteraction(f'rotation_{index}', target=f'unit_{index}') for index in range(self.n_units)]
        self.system_ = System(self.label, elements=[*units, *axons], interactions=[*rotations, *adaptations])
