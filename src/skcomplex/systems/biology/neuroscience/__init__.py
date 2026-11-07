"""Neuroscience: spiking and conductance-based neurons, populations, networks, excitable networks, and synapses."""

from ._associative import AssociativeSiteElement, CliqueCompetitionInteraction, associative_thought_process
from ._conductances import FitzHughNagumoNeuronElement, HodgkinHuxleyElement, fitzhugh_nagumo_ode, hodgkin_huxley
from ._excitable import ExcitableElement, kinouchi_copelli
from ._networks import (
    DelayedSynapseInteraction,
    GlobalInhibitionInteraction,
    InhibitorDriveInteraction,
    InhibitorElement,
    SpikingNeuronElement,
    TermanWangElement,
    delayed_lif_network,
    legion,
)
from ._neural_fields import NeuralFieldSiteElement, amari_neural_field
from ._neural_masses import (
    NeuralMassElement,
    SynapticDriveInteraction,
    ThetaForcingInteraction,
    next_generation_neural_mass,
)
from ._neurons import IzhikevichElement, LeakyIntegrateAndFireElement, integrate_and_fire, izhikevich
from ._plastic_networks import (
    AxonElement,
    DelayAdaptationInteraction,
    PhaseUnitElement,
    SpikeTimingPlasticityInteraction,
    SynapseElement,
    TracedThetaNeuronElement,
    adaptive_delay_network,
    stdp_network,
)
from ._populations import (
    MutualInhibitionInteraction,
    PolyhomeostaticNeuronElement,
    PulseCouplingInteraction,
    RateUnitElement,
    WilsonCowanElement,
    polyhomeostatic_neuron,
    theta_neurons,
    wilson_cowan,
    winners_take_all,
)
from ._spike_reset import SpikeResetInteraction
from ._synapses import AdaptrodeElement, adaptrode
from ._theta_neuron import ThetaNeuronElement

__all__: list[str] = [
    'AdaptrodeElement',
    'AssociativeSiteElement',
    'AxonElement',
    'CliqueCompetitionInteraction',
    'DelayAdaptationInteraction',
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
    'NeuralMassElement',
    'PhaseUnitElement',
    'PolyhomeostaticNeuronElement',
    'PulseCouplingInteraction',
    'RateUnitElement',
    'SpikeResetInteraction',
    'SpikeTimingPlasticityInteraction',
    'SpikingNeuronElement',
    'SynapseElement',
    'SynapticDriveInteraction',
    'TermanWangElement',
    'ThetaForcingInteraction',
    'ThetaNeuronElement',
    'TracedThetaNeuronElement',
    'WilsonCowanElement',
    'adaptive_delay_network',
    'adaptrode',
    'amari_neural_field',
    'associative_thought_process',
    'delayed_lif_network',
    'fitzhugh_nagumo_ode',
    'hodgkin_huxley',
    'integrate_and_fire',
    'izhikevich',
    'kinouchi_copelli',
    'legion',
    'next_generation_neural_mass',
    'polyhomeostatic_neuron',
    'stdp_network',
    'theta_neurons',
    'wilson_cowan',
    'winners_take_all',
]
