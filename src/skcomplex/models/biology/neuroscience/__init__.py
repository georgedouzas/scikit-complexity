"""Neuroscience: spiking and conductance-based neurons, populations, networks, excitable networks, and synapses."""

from ._conductances import FitzHughNagumoNeuronComponent, HodgkinHuxleyComponent, fitzhugh_nagumo_ode, hodgkin_huxley
from ._excitable import ExcitableElementComponent, kinouchi_copelli
from ._networks import (
    DelayedSynapseInteraction,
    GlobalInhibitionInteraction,
    InhibitorComponent,
    SpikingNeuronComponent,
    TermanWangComponent,
    delayed_lif_network,
    legion,
)
from ._neurons import IzhikevichComponent, LeakyIntegrateAndFireComponent, integrate_and_fire, izhikevich
from ._populations import (
    MutualInhibitionInteraction,
    PolyhomeostaticNeuronComponent,
    PulseCouplingInteraction,
    RateUnitComponent,
    ThetaNeuronComponent,
    WilsonCowanComponent,
    polyhomeostatic_neuron,
    theta_neurons,
    wilson_cowan,
    winners_take_all,
)
from ._synapses import AdaptrodeComponent, adaptrode

__all__: list[str] = [
    'AdaptrodeComponent',
    'DelayedSynapseInteraction',
    'ExcitableElementComponent',
    'FitzHughNagumoNeuronComponent',
    'GlobalInhibitionInteraction',
    'HodgkinHuxleyComponent',
    'InhibitorComponent',
    'IzhikevichComponent',
    'LeakyIntegrateAndFireComponent',
    'MutualInhibitionInteraction',
    'PolyhomeostaticNeuronComponent',
    'PulseCouplingInteraction',
    'RateUnitComponent',
    'SpikingNeuronComponent',
    'TermanWangComponent',
    'ThetaNeuronComponent',
    'WilsonCowanComponent',
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
