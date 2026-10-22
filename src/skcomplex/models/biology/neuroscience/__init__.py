"""Neuroscience: spiking and conductance-based neurons, populations, networks, excitable networks, and synapses."""

from ._conductances import FitzHughNagumoNeuron, HodgkinHuxley, fitzhugh_nagumo_ode, hodgkin_huxley
from ._excitable import ExcitableElement, kinouchi_copelli
from ._networks import (
    DelayedSynapse,
    GlobalInhibition,
    Inhibitor,
    SpikingNeuron,
    TermanWang,
    delayed_lif_network,
    legion,
)
from ._neurons import Izhikevich, LeakyIntegrateAndFire, integrate_and_fire, izhikevich
from ._populations import (
    MutualInhibition,
    PolyhomeostaticNeuron,
    PulseCoupling,
    RateUnit,
    ThetaNeuron,
    WilsonCowan,
    polyhomeostatic_neuron,
    theta_neurons,
    wilson_cowan,
    winners_take_all,
)
from ._synapses import Adaptrode, adaptrode

__all__: list[str] = [
    'Adaptrode',
    'DelayedSynapse',
    'ExcitableElement',
    'FitzHughNagumoNeuron',
    'GlobalInhibition',
    'HodgkinHuxley',
    'Inhibitor',
    'Izhikevich',
    'LeakyIntegrateAndFire',
    'MutualInhibition',
    'PolyhomeostaticNeuron',
    'PulseCoupling',
    'RateUnit',
    'SpikingNeuron',
    'TermanWang',
    'ThetaNeuron',
    'WilsonCowan',
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
