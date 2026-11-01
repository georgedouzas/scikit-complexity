"""Neuroscience: spiking and conductance-based neurons, populations, networks, excitable networks, and synapses."""

from ._conductances import fitzhugh_nagumo_ode, hodgkin_huxley
from ._delayed_synapse import DelayedSynapseInteraction
from ._excitable import ExcitableElement, kinouchi_copelli
from ._fitz_hugh_nagumo_neuron import FitzHughNagumoNeuronElement
from ._global_inhibition import GlobalInhibitionInteraction
from ._hodgkin_huxley import HodgkinHuxleyElement
from ._inhibitor import InhibitorElement
from ._inhibitor_drive import InhibitorDriveInteraction
from ._izhikevich import IzhikevichElement
from ._leaky_integrate_and_fire import LeakyIntegrateAndFireElement
from ._mutual_inhibition import MutualInhibitionInteraction
from ._networks import delayed_lif_network, legion
from ._neural_field_site import NeuralFieldSiteElement
from ._neural_fields import amari_neural_field
from ._neurons import integrate_and_fire, izhikevich
from ._polyhomeostatic_neuron import PolyhomeostaticNeuronElement
from ._populations import polyhomeostatic_neuron, theta_neurons, wilson_cowan, winners_take_all
from ._pulse_coupling import PulseCouplingInteraction
from ._rate_unit import RateUnitElement
from ._spike_reset import SpikeResetInteraction
from ._spiking_neuron import SpikingNeuronElement
from ._synapses import AdaptrodeElement, adaptrode
from ._terman_wang import TermanWangElement
from ._theta_neuron import ThetaNeuronElement
from ._wilson_cowan import WilsonCowanElement

__all__: list[str] = [
    'AdaptrodeElement',
    'DelayedSynapseInteraction',
    'ExcitableElement',
    'FitzHughNagumoNeuronElement',
    'GlobalInhibitionInteraction',
    'HodgkinHuxleyElement',
    'InhibitorDriveInteraction',
    'InhibitorElement',
    'IzhikevichElement',
    'LeakyIntegrateAndFireElement',
    'MutualInhibitionInteraction',
    'NeuralFieldSiteElement',
    'PolyhomeostaticNeuronElement',
    'PulseCouplingInteraction',
    'RateUnitElement',
    'SpikeResetInteraction',
    'SpikingNeuronElement',
    'TermanWangElement',
    'ThetaNeuronElement',
    'WilsonCowanElement',
    'adaptrode',
    'amari_neural_field',
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
