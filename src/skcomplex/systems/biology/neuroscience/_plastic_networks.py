"""Neuroscience systems: networks whose synapses adapt to the activity of the neurons they join."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from itertools import permutations

import networkx as nx
import numpy as np

from ....core.components.elements import GroupElement, LatticeElement
from ....core.components.spaces import NetworkSpace
from ....core.components.system import System
from ..._catalog import register_system
from ._axon import AxonElement
from ._delay_adaptation import DelayAdaptationInteraction
from ._phase_unit import PhaseUnitElement
from ._spike_timing_plasticity import SpikeTimingPlasticityInteraction
from ._synapse import SynapseElement
from ._traced_theta_neuron import TracedThetaNeuronElement


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
