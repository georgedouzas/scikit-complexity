"""Neuroscience: spiking and conductance-based neurons, populations, networks, excitable networks, and synapses."""

from ._conductances import FitzHughNagumoNeuronElement, HodgkinHuxleyElement, fitzhugh_nagumo_ode, hodgkin_huxley
from ._excitable import ExcitableElementElement, kinouchi_copelli
from ._networks import (
    DelayedSynapseInteraction,
    GlobalInhibitionInteraction,
    InhibitorElement,
    SpikingNeuronElement,
    TermanWangElement,
    delayed_lif_network,
    legion,
)
from ._neurons import IzhikevichElement, LeakyIntegrateAndFireElement, integrate_and_fire, izhikevich
from ._populations import (
    MutualInhibitionInteraction,
    PolyhomeostaticNeuronElement,
    PulseCouplingInteraction,
    RateUnitElement,
    ThetaNeuronElement,
    WilsonCowanElement,
    polyhomeostatic_neuron,
    theta_neurons,
    wilson_cowan,
    winners_take_all,
)
from ._synapses import AdaptrodeElement, adaptrode

__all__: list[str] = [
    'AdaptrodeElement',
    'DelayedSynapseInteraction',
    'ExcitableElementElement',
    'FitzHughNagumoNeuronElement',
    'GlobalInhibitionInteraction',
    'HodgkinHuxleyElement',
    'InhibitorElement',
    'IzhikevichElement',
    'LeakyIntegrateAndFireElement',
    'MutualInhibitionInteraction',
    'PolyhomeostaticNeuronElement',
    'PulseCouplingInteraction',
    'RateUnitElement',
    'SpikingNeuronElement',
    'TermanWangElement',
    'ThetaNeuronElement',
    'WilsonCowanElement',
    'adaptrode',
    'delayed_lif_network',
    'fitzhugh_nagumo_ode',
    'hodgkin_huxley',
    'integrate_and_fire',
    'izhikevich',
    'kinouchi_copelli',
    'legion',
    'polyhomeostatic_neuron',
    'theta_neurons',
    'wilson_cowan',
    'winners_take_all',
]
