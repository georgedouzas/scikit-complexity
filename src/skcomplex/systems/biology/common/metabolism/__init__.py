"""Metabolism shared by the biology packages.

It covers metabolic networks, their standard formats, and their analyses.
"""

from ._analyses import (
    CobraMeasuresResult,
    FluxVariabilityResult,
    ProductionScanResult,
    StrainDesignResult,
    cobra_measures,
    flux_variability,
    production_scan,
    strain_design,
)
from ._cobra_io import from_cobra, load_extra, load_network, save_network, to_cobra
from ._community import CommunityGrowthResult, community_growth
from ._culture import (
    BiomassElement,
    CommunityInteraction,
    CommunityMediumElement,
    ExchangeInteraction,
    GrowthInteraction,
    ObjectiveFloorInteraction,
    SubstrateElement,
    SwitchInteraction,
    community_medium_type,
)
from ._metabolic_network import MetabolicNetworkElement, metabolic_network_type
from ._reaction_spec import ReactionSpec
from ._toy_network import ToyNetworkElement

__all__: list[str] = [
    'BiomassElement',
    'CobraMeasuresResult',
    'CommunityGrowthResult',
    'CommunityInteraction',
    'CommunityMediumElement',
    'ExchangeInteraction',
    'FluxVariabilityResult',
    'GrowthInteraction',
    'MetabolicNetworkElement',
    'ObjectiveFloorInteraction',
    'ProductionScanResult',
    'ReactionSpec',
    'StrainDesignResult',
    'SubstrateElement',
    'SwitchInteraction',
    'ToyNetworkElement',
    'cobra_measures',
    'community_growth',
    'community_medium_type',
    'flux_variability',
    'from_cobra',
    'load_extra',
    'load_network',
    'metabolic_network_type',
    'production_scan',
    'save_network',
    'strain_design',
    'to_cobra',
]
