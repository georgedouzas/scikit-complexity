"""Neuroscience systems.

They cover spiking and conductance-based neurons, populations, networks, excitable networks, and synapses.
"""

from ..common.excitable import SpikeResetInteraction, ThetaNeuronElement, ThetaNeuronInteraction
from ._associative import (
    AssociativeSiteElement,
    AssociativeThoughtProcessSystem,
    CliqueCompetitionInteraction,
    ReservoirInteraction,
)
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
    InhibitorDecayInteraction,
    InhibitorDriveInteraction,
    InhibitorElement,
    LegionSystem,
    SpikingNeuronElement,
    TermanWangElement,
    TermanWangInteraction,
)
from ._neural_fields import AmariNeuralFieldSystem, NeuralFieldSiteElement
from ._neural_masses import (
    NeuralMassElement,
    NeuralMassInteraction,
    NextGenerationNeuralMassSystem,
    SynapticDriveInteraction,
    ThetaForcingInteraction,
)
from ._neurons import (
    IntegrateAndFireSystem,
    IzhikevichElement,
    IzhikevichInteraction,
    IzhikevichSystem,
    LeakyIntegrateAndFireElement,
    LeakyIntegrateAndFireInteraction,
)
from ._plastic_networks import (
    AdaptiveDelayNetworkSystem,
    AxonElement,
    DelayAdaptationInteraction,
    PhaseUnitElement,
    RotationInteraction,
    SpikeTimingPlasticityInteraction,
    SpikeTraceInteraction,
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
    SaturatingGrowthInteraction,
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
    'InhibitorDecayInteraction',
    'InhibitorDriveInteraction',
    'InhibitorElement',
    'IntegrateAndFireSystem',
    'IzhikevichElement',
    'IzhikevichInteraction',
    'IzhikevichSystem',
    'KinouchiCopelliSystem',
    'LeakyIntegrateAndFireElement',
    'LeakyIntegrateAndFireInteraction',
    'LeakyIntegrationInteraction',
    'LegionSystem',
    'MutualInhibitionInteraction',
    'NeuralFieldSiteElement',
    'NeuralMassElement',
    'NeuralMassInteraction',
    'NextGenerationNeuralMassSystem',
    'PhaseUnitElement',
    'PolyhomeostaticNeuronElement',
    'PolyhomeostaticNeuronSystem',
    'PulseCouplingInteraction',
    'RateUnitElement',
    'ReservoirInteraction',
    'RotationInteraction',
    'STDPNetworkSystem',
    'SaturatingGrowthInteraction',
    'SpikeResetInteraction',
    'SpikeTimingPlasticityInteraction',
    'SpikeTraceInteraction',
    'SpikingNeuronElement',
    'SynapseElement',
    'SynapticDriveInteraction',
    'TermanWangElement',
    'TermanWangInteraction',
    'ThetaForcingInteraction',
    'ThetaNeuronElement',
    'ThetaNeuronInteraction',
    'ThetaNeuronsSystem',
    'TracedThetaNeuronElement',
    'WilsonCowanElement',
    'WilsonCowanSystem',
    'WinnersTakeAllSystem',
]
