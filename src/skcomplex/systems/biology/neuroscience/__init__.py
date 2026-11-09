"""Neuroscience systems.

They cover spiking and conductance-based neurons, populations, networks, excitable networks, and synapses.
"""

from ..common.excitable import SpikeResetInteraction, ThetaNeuronElement
from ._associative import AssociativeSiteElement, AssociativeThoughtProcessSystem, CliqueCompetitionInteraction
from ._conductances import (
    FitzHughNagumoNeuronElement,
    FitzhughNagumoODESystem,
    HodgkinHuxleyElement,
    HodgkinHuxleySystem,
)
from ._excitable import ExcitableElement, KinouchiCopelliSystem
from ._leaky_integration import LeakyIntegrationInteraction
from ._networks import (
    DelayedLIFNetworkSystem,
    DelayedSynapseInteraction,
    GlobalInhibitionInteraction,
    InhibitorDriveInteraction,
    InhibitorElement,
    LegionSystem,
    SpikingNeuronElement,
    TermanWangElement,
)
from ._neural_fields import AmariNeuralFieldSystem, NeuralFieldSiteElement
from ._neural_masses import (
    NeuralMassElement,
    NextGenerationNeuralMassSystem,
    SynapticDriveInteraction,
    ThetaForcingInteraction,
)
from ._neurons import IntegrateAndFireSystem, IzhikevichElement, IzhikevichSystem, LeakyIntegrateAndFireElement
from ._plastic_networks import (
    AdaptiveDelayNetworkSystem,
    AxonElement,
    DelayAdaptationInteraction,
    PhaseUnitElement,
    SpikeTimingPlasticityInteraction,
    STDPNetworkSystem,
    SynapseElement,
    TracedThetaNeuronElement,
)
from ._populations import (
    MutualInhibitionInteraction,
    PolyhomeostaticNeuronElement,
    PolyhomeostaticNeuronSystem,
    PulseCouplingInteraction,
    RateUnitElement,
    ThetaNeuronsSystem,
    WilsonCowanElement,
    WilsonCowanSystem,
    WinnersTakeAllSystem,
)
from ._synapses import AdaptrodeSystem

__all__: list[str] = [
    'AdaptiveDelayNetworkSystem',
    'AdaptrodeSystem',
    'AmariNeuralFieldSystem',
    'AssociativeSiteElement',
    'AssociativeThoughtProcessSystem',
    'AxonElement',
    'CliqueCompetitionInteraction',
    'DelayAdaptationInteraction',
    'DelayedLIFNetworkSystem',
    'DelayedSynapseInteraction',
    'ExcitableElement',
    'FitzHughNagumoNeuronElement',
    'FitzhughNagumoODESystem',
    'GlobalInhibitionInteraction',
    'HodgkinHuxleyElement',
    'HodgkinHuxleySystem',
    'InhibitorDriveInteraction',
    'InhibitorElement',
    'IntegrateAndFireSystem',
    'IzhikevichElement',
    'IzhikevichSystem',
    'KinouchiCopelliSystem',
    'LeakyIntegrateAndFireElement',
    'LeakyIntegrationInteraction',
    'LegionSystem',
    'MutualInhibitionInteraction',
    'NeuralFieldSiteElement',
    'NeuralMassElement',
    'NextGenerationNeuralMassSystem',
    'PhaseUnitElement',
    'PolyhomeostaticNeuronElement',
    'PolyhomeostaticNeuronSystem',
    'PulseCouplingInteraction',
    'RateUnitElement',
    'STDPNetworkSystem',
    'SpikeResetInteraction',
    'SpikeTimingPlasticityInteraction',
    'SpikingNeuronElement',
    'SynapseElement',
    'SynapticDriveInteraction',
    'TermanWangElement',
    'ThetaForcingInteraction',
    'ThetaNeuronElement',
    'ThetaNeuronsSystem',
    'TracedThetaNeuronElement',
    'WilsonCowanElement',
    'WilsonCowanSystem',
    'WinnersTakeAllSystem',
]
