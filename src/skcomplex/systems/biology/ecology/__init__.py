"""Ecology systems.

They cover population growth, predation, competition, food webs, succession, grazing, and neutral communities.
"""

from ..common.populations import PredationInteraction
from ._agent_predation import AgentBasedLotkaVolterraSystem, PredatorElement, PreyElement
from ._cell_quota import DroopCellQuotaSystem
from ._clustering import HabitatPatchElement, HoppingInteraction, NeutralClusteringSystem
from ._comets import CometsSystem
from ._communities import (
    CompetitionInteraction,
    CompetitionSystem,
    FoodWebsSystem,
    MutualismInteraction,
    MutualismSystem,
)
from ._community_fba import CommunityFBASystem
from ._consumer_resources import (
    ConsumerElement,
    ConsumerGrowthInteraction,
    ConsumerResourceSystem,
    ConsumptionInteraction,
    ResourceElement,
    ResourceTurnoverInteraction,
)
from ._dynamic_multispecies_fba import DynamicMultispeciesFBASystem
from ._factories import (
    ExponentialGrowthSystem,
    FibonacciRabbitsSystem,
    HutchinsonSystem,
    LogisticGrowthSystem,
    LotkaVolterraSystem,
    PenaeidShrimpSystem,
    RickerGrowthInteraction,
    RickerSystem,
    SeasonalLogisticGrowthInteraction,
    VaryingCapacityLogisticSystem,
)
from ._forest_size_structure import ForestSizeStructureSystem, SizeGrowthInteraction
from ._fronts import FisherKPPElement, FisherKPPSystem
from ._higher_order import HigherOrderRandomLotkaVolterraSystem, TripletInteraction
from ._instability import IntrinsicInstabilitySystem, UnstablePopulationsElement
from ._meadow import CowElement, GrazingInteraction, HerdingInteraction, MeadowCapacitySystem, PastureElement
from ._microbial_communities import GeneralizedLotkaVolterraInteraction, GeneralizedLotkaVolterraSystem, MicrobeElement
from ._neutral import NeutralMacroecologySystem, NeutralSpeciesElement
from ._predation import (
    LotkaVolterraWithHarvestingSystem,
    NicholsonBaileySystem,
    ParasitismInteraction,
    PredatorPreyEventsInteraction,
    RosenzweigMacArthurInteraction,
    RosenzweigMacarthurSystem,
    StochasticPredatorPreySystem,
)
from ._sequential import AbundanceElement, LinearGrowthInteraction, SequentiallyLinearEcosystemSystem
from ._speciation import NeutralBiodiversitySystem, NeutralReplacementInteraction, OrganismElement
from ._stigmergy import (
    EmotionMarksElement,
    ForagerElement,
    PlantElement,
    StigmergicPredatorElement,
    StigmergicPredatorPreySystem,
    StigmergicPreyElement,
)
from ._succession import MangroveSuccessionSystem, PioneerGrowthInteraction, StandElement, SuccessionalGrowthInteraction
from ._wolf_sheep import GrassElement, SheepElement, WolfElement, WolfSheepSystem

__all__: list[str] = [
    'AbundanceElement',
    'AgentBasedLotkaVolterraSystem',
    'CometsSystem',
    'CommunityFBASystem',
    'CompetitionInteraction',
    'CompetitionSystem',
    'ConsumerElement',
    'ConsumerGrowthInteraction',
    'ConsumerResourceSystem',
    'ConsumptionInteraction',
    'CowElement',
    'DroopCellQuotaSystem',
    'DynamicMultispeciesFBASystem',
    'EmotionMarksElement',
    'ExponentialGrowthSystem',
    'FibonacciRabbitsSystem',
    'FisherKPPElement',
    'FisherKPPSystem',
    'FoodWebsSystem',
    'ForagerElement',
    'ForestSizeStructureSystem',
    'GeneralizedLotkaVolterraInteraction',
    'GeneralizedLotkaVolterraSystem',
    'GrassElement',
    'GrazingInteraction',
    'HabitatPatchElement',
    'HerdingInteraction',
    'HigherOrderRandomLotkaVolterraSystem',
    'HoppingInteraction',
    'HutchinsonSystem',
    'IntrinsicInstabilitySystem',
    'LinearGrowthInteraction',
    'LogisticGrowthSystem',
    'LotkaVolterraSystem',
    'LotkaVolterraWithHarvestingSystem',
    'MangroveSuccessionSystem',
    'MeadowCapacitySystem',
    'MicrobeElement',
    'MutualismInteraction',
    'MutualismSystem',
    'NeutralBiodiversitySystem',
    'NeutralClusteringSystem',
    'NeutralMacroecologySystem',
    'NeutralReplacementInteraction',
    'NeutralSpeciesElement',
    'NicholsonBaileySystem',
    'OrganismElement',
    'ParasitismInteraction',
    'PastureElement',
    'PenaeidShrimpSystem',
    'PioneerGrowthInteraction',
    'PlantElement',
    'PredationInteraction',
    'PredatorElement',
    'PredatorPreyEventsInteraction',
    'PreyElement',
    'ResourceElement',
    'ResourceTurnoverInteraction',
    'RickerGrowthInteraction',
    'RickerSystem',
    'RosenzweigMacArthurInteraction',
    'RosenzweigMacarthurSystem',
    'SeasonalLogisticGrowthInteraction',
    'SequentiallyLinearEcosystemSystem',
    'SheepElement',
    'SizeGrowthInteraction',
    'StandElement',
    'StigmergicPredatorElement',
    'StigmergicPredatorPreySystem',
    'StigmergicPreyElement',
    'StochasticPredatorPreySystem',
    'SuccessionalGrowthInteraction',
    'TripletInteraction',
    'UnstablePopulationsElement',
    'VaryingCapacityLogisticSystem',
    'WolfElement',
    'WolfSheepSystem',
]
