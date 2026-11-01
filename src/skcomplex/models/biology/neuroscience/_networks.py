"""Neuroscience models: networks of spiking neurons and of relaxation oscillators."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import networkx as nx
import numpy as np
from scipy.ndimage import convolve

from ....core.elements import FieldElement
from ....core.interactions import RateInteraction, RelaxationInteraction
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....types import Array
from ..._catalog import register_model
from ._delayed_synapse import DelayedSynapseInteraction
from ._global_inhibition import GlobalInhibitionInteraction
from ._inhibitor import InhibitorElement
from ._inhibitor_drive import InhibitorDriveInteraction
from ._spike_reset import SpikeResetInteraction
from ._spiking_neuron import SpikingNeuronElement
from ._terman_wang import SILENT, TermanWangElement

NEAREST = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


LEGION_OBJECTS = ((1, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 1))


THRESHOLD = 20.0


RESET = 10.0


TRACE_TIME = 2.0


def _random_connections(n_neurons: int, probability: float, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Return the directed links of a random network without self-links."""
    return list(nx.gnp_random_graph(n_neurons, probability, seed=rng, directed=True).edges)


def _link_weights(stimulus: Array, total: float) -> Array:
    """Return the weight of each link into every site, a total weight shared among its stimulated neighbours."""
    neighbours = convolve(stimulus, NEAREST, mode='constant')
    return total / np.maximum(neighbours, 1.0)


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
            The model, with the potentials `neuron_<i>.v`, the traces `neuron_<i>.s`, and the spikes of neuron `i` as
            `spike_<i>` events.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import delayed_lif_network
        >>> len(delayed_lif_network(5).elements)
        5
    """
    rng = np.random.default_rng(random_state)
    potentials = rng.uniform(10.0, 20.0, n_neurons)
    neurons = [
        SpikingNeuronElement(f'neuron_{index}', v=float(potentials[index]), mu=mu, tau_s=TRACE_TIME)
        for index in range(n_neurons)
    ]
    leaks = [
        RelaxationInteraction(
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
            J=J,
            D=D,
        )
        for source, target in _random_connections(n_neurons, probability, rng)
    ]
    return System(elements=neurons, interactions=[*leaks, *traces, *spikes, *synapses])


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
    drive = InhibitorDriveInteraction('drive', inhibitor='inhibitor', oscillators='oscillators')
    inhibition = GlobalInhibitionInteraction('inhibition', inhibitor='inhibitor', oscillators='oscillators', W_z=W_z)
    return System(elements=[oscillators, InhibitorElement('inhibitor')], interactions=[drive, inhibition])
