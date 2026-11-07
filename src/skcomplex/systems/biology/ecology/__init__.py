"""Ecology: population growth, predation, competition, food webs, succession, grazing, and neutral communities."""

from ._agent_predation import PredatorElement, PreyElement, agent_based_lotka_volterra
from ._cell_quota import QuotaPhytoplanktonElement, droop_cell_quota
from ._clustering import HabitatPatchElement, HoppingInteraction, neutral_clustering
from ._comets import comets
from ._communities import CompetitionInteraction, MutualismInteraction, competition, food_webs, mutualism
from ._community_fba import community_fba
from ._consumer_resources import (
    ConsumerElement,
    ConsumerGrowthInteraction,
    ConsumptionInteraction,
    ResourceElement,
    ResourceTurnoverInteraction,
    consumer_resource,
)
from ._dynamic_multispecies_fba import dynamic_multispecies_fba
from ._factories import (
    RabbitsElement,
    RickerGrowthInteraction,
    SeasonalLogisticGrowthInteraction,
    ShrimpElement,
    exponential_growth,
    fibonacci_rabbits,
    hutchinson,
    logistic_growth,
    lotka_volterra,
    penaeid_shrimp,
    ricker,
    varying_capacity_logistic,
)
from ._forest_size_structure import forest_size_structure
from ._fronts import FisherKPPElement, fisher_kpp
from ._higher_order import TripletInteraction, higher_order_random_lotka_volterra
from ._instability import UnstablePopulationsElement, intrinsic_instability
from ._meadow import CowElement, GrazingInteraction, HerdingInteraction, PastureElement, meadow_capacity
from ._microbial_communities import GeneralizedLotkaVolterraInteraction, MicrobeElement, generalized_lotka_volterra
from ._neutral import NeutralSpeciesElement, neutral_macroecology
from ._predation import (
    ParasitismInteraction,
    PredatorPreyEventsInteraction,
    RosenzweigMacArthurInteraction,
    lotka_volterra_with_harvesting,
    nicholson_bailey,
    rosenzweig_macarthur,
    stochastic_predator_prey,
)
from ._predation_interaction import PredationInteraction
from ._sequential import AbundanceElement, LinearGrowthInteraction, sequentially_linear_ecosystem
from ._speciation import NeutralReplacementInteraction, OrganismElement, neutral_biodiversity
from ._stigmergy import (
    EmotionMarksElement,
    ForagerElement,
    PlantElement,
    StigmergicPredatorElement,
    StigmergicPreyElement,
    stigmergic_predator_prey,
)
from ._succession import PioneerGrowthInteraction, StandElement, SuccessionalGrowthInteraction, mangrove_succession
from ._wolf_sheep import GrassElement, SheepElement, WolfElement, wolf_sheep

__all__: list[str] = [
    'AbundanceElement',
    'CompetitionInteraction',
    'ConsumerElement',
    'ConsumerGrowthInteraction',
    'ConsumptionInteraction',
    'CowElement',
    'EmotionMarksElement',
    'FisherKPPElement',
    'ForagerElement',
    'GeneralizedLotkaVolterraInteraction',
    'GrassElement',
    'GrazingInteraction',
    'HabitatPatchElement',
    'HerdingInteraction',
    'HoppingInteraction',
    'LinearGrowthInteraction',
    'MicrobeElement',
    'MutualismInteraction',
    'NeutralReplacementInteraction',
    'NeutralSpeciesElement',
    'OrganismElement',
    'ParasitismInteraction',
    'PastureElement',
    'PioneerGrowthInteraction',
    'PlantElement',
    'PredationInteraction',
    'PredatorElement',
    'PredatorPreyEventsInteraction',
    'PreyElement',
    'QuotaPhytoplanktonElement',
    'RabbitsElement',
    'ResourceElement',
    'ResourceTurnoverInteraction',
    'RickerGrowthInteraction',
    'RosenzweigMacArthurInteraction',
    'SeasonalLogisticGrowthInteraction',
    'SheepElement',
    'ShrimpElement',
    'StandElement',
    'StigmergicPredatorElement',
    'StigmergicPreyElement',
    'SuccessionalGrowthInteraction',
    'TripletInteraction',
    'UnstablePopulationsElement',
    'WolfElement',
    'agent_based_lotka_volterra',
    'comets',
    'community_fba',
    'competition',
    'consumer_resource',
    'droop_cell_quota',
    'dynamic_multispecies_fba',
    'exponential_growth',
    'fibonacci_rabbits',
    'fisher_kpp',
    'food_webs',
    'forest_size_structure',
    'generalized_lotka_volterra',
    'higher_order_random_lotka_volterra',
    'hutchinson',
    'intrinsic_instability',
    'logistic_growth',
    'lotka_volterra',
    'lotka_volterra_with_harvesting',
    'mangrove_succession',
    'meadow_capacity',
    'mutualism',
    'neutral_biodiversity',
    'neutral_clustering',
    'neutral_macroecology',
    'nicholson_bailey',
    'penaeid_shrimp',
    'ricker',
    'rosenzweig_macarthur',
    'sequentially_linear_ecosystem',
    'stigmergic_predator_prey',
    'stochastic_predator_prey',
    'varying_capacity_logistic',
    'wolf_sheep',
]
