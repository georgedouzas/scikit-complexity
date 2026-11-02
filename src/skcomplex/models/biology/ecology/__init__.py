"""Ecology: population growth, predation, competition, food webs, succession, grazing, and neutral communities."""

from ._abundance import AbundanceElement
from ._agent_predation import agent_based_lotka_volterra
from ._cell_quota import QuotaPhytoplanktonElement, droop_cell_quota
from ._clustering import neutral_clustering
from ._communities import competition, food_webs, mutualism
from ._competition import CompetitionInteraction
from ._cow import CowElement
from ._emotion_marks import EmotionMarksElement
from ._factories import (
    RabbitsElement,
    exponential_growth,
    fibonacci_rabbits,
    hutchinson,
    logistic_growth,
    lotka_volterra,
    varying_capacity_logistic,
)
from ._forager import ForagerElement
from ._fronts import FisherKPPElement, fisher_kpp
from ._generalized_lotka_volterra import GeneralizedLotkaVolterraInteraction
from ._grass import GrassElement
from ._grazing import GrazingInteraction
from ._habitat_patch import HabitatPatchElement
from ._herding import HerdingInteraction
from ._higher_order import higher_order_random_lotka_volterra
from ._hopping import HoppingInteraction
from ._instability import UnstablePopulationsElement, intrinsic_instability
from ._linear_growth import LinearGrowthInteraction
from ._meadow import meadow_capacity
from ._microbe import MicrobeElement
from ._microbial_communities import generalized_lotka_volterra
from ._mutualism import MutualismInteraction
from ._neutral import NeutralSpeciesElement, neutral_macroecology
from ._neutral_replacement import NeutralReplacementInteraction
from ._organism import OrganismElement
from ._parasitism import ParasitismInteraction
from ._pasture import PastureElement
from ._pioneer_growth import PioneerGrowthInteraction
from ._plant import PlantElement
from ._predation import lotka_volterra_with_harvesting, nicholson_bailey, rosenzweig_macarthur, stochastic_predator_prey
from ._predation_interaction import PredationInteraction
from ._predator import PredatorElement
from ._predator_prey_events import PredatorPreyEventsInteraction
from ._prey import PreyElement
from ._rosenzweig_macarthur import RosenzweigMacArthurInteraction
from ._seasonal_logistic_growth import SeasonalLogisticGrowthInteraction
from ._sequential import sequentially_linear_ecosystem
from ._sheep import SheepElement
from ._speciation import neutral_biodiversity
from ._stand import StandElement
from ._stigmergic_predator import StigmergicPredatorElement
from ._stigmergic_prey import StigmergicPreyElement
from ._stigmergy import stigmergic_predator_prey
from ._succession import mangrove_succession
from ._successional_growth import SuccessionalGrowthInteraction
from ._triplet import TripletInteraction
from ._wolf import WolfElement
from ._wolf_sheep import wolf_sheep

__all__: list[str] = [
    'AbundanceElement',
    'CompetitionInteraction',
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
    'RosenzweigMacArthurInteraction',
    'SeasonalLogisticGrowthInteraction',
    'SheepElement',
    'StandElement',
    'StigmergicPredatorElement',
    'StigmergicPreyElement',
    'SuccessionalGrowthInteraction',
    'TripletInteraction',
    'UnstablePopulationsElement',
    'WolfElement',
    'agent_based_lotka_volterra',
    'competition',
    'droop_cell_quota',
    'exponential_growth',
    'fibonacci_rabbits',
    'fisher_kpp',
    'food_webs',
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
    'rosenzweig_macarthur',
    'sequentially_linear_ecosystem',
    'stigmergic_predator_prey',
    'stochastic_predator_prey',
    'varying_capacity_logistic',
    'wolf_sheep',
]
