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
    ExchangeInteraction,
    GrowthInteraction,
    ObjectiveFloorInteraction,
    SubstrateElement,
    SwitchInteraction,
)
from ._flux_balance import FluxBalanceInteraction, MetabolicCellElement
from ._reaction_spec import ReactionSpec
from ._toy_network import ToyFluxBalanceInteraction

__all__: list[str] = [
    'BiomassElement',
    'CobraMeasuresResult',
    'CommunityGrowthResult',
    'CommunityInteraction',
    'ExchangeInteraction',
    'FluxBalanceInteraction',
    'FluxVariabilityResult',
    'GrowthInteraction',
    'MetabolicCellElement',
    'ObjectiveFloorInteraction',
    'ProductionScanResult',
    'ReactionSpec',
    'StrainDesignResult',
    'SubstrateElement',
    'SwitchInteraction',
    'ToyFluxBalanceInteraction',
    'cobra_measures',
    'community_growth',
    'flux_variability',
    'from_cobra',
    'load_extra',
    'load_network',
    'production_scan',
    'save_network',
    'strain_design',
    'to_cobra',
]
