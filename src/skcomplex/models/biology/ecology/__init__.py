"""Ecology: population growth, predation, competition, food webs, succession, grazing, and neutral communities."""

from ._agent_predation import PredatorComponent, PreyComponent, agent_based_lotka_volterra
from ._cell_quota import QuotaPhytoplanktonComponent, droop_cell_quota
from ._clustering import HoppingInteraction, PatchComponent, neutral_clustering
from ._communities import CompetitionInteraction, MutualismInteraction, competition, food_webs, mutualism
from ._factories import (
    RabbitsComponent,
    exponential_growth,
    fibonacci_rabbits,
    hutchinson,
    logistic_growth,
    lotka_volterra,
    varying_capacity_logistic,
)
from ._fronts import FisherKPPComponent, fisher_kpp
from ._instability import UnstablePopulationsComponent, intrinsic_instability
from ._meadow import CowComponent, GrazingInteraction, HerdingInteraction, PastureComponent, meadow_capacity
from ._microbial_communities import GeneralizedLotkaVolterraInteraction, MicrobeComponent, generalized_lotka_volterra
from ._neutral import NeutralSpeciesComponent, neutral_macroecology
from ._populations import (
    DelayedLogisticGrowthInteraction,
    ExponentialGrowthInteraction,
    LogisticGrowthInteraction,
    MortalityInteraction,
    PopulationComponent,
    PredationInteraction,
    SeasonalLogisticGrowthInteraction,
)
from ._predation import (
    ParasitismInteraction,
    PredatorPreyEventsInteraction,
    RosenzweigMacArthurInteraction,
    lotka_volterra_with_harvesting,
    nicholson_bailey,
    rosenzweig_macarthur,
    stochastic_predator_prey,
)
from ._sequential import (
    AbundanceComponent,
    InfluenceComponent,
    InfluenceDrawInteraction,
    LinearGrowthInteraction,
    sequentially_linear_ecosystem,
)
from ._speciation import NeutralReplacementInteraction, OrganismComponent, neutral_biodiversity
from ._stigmergy import (
    EmotionMarksComponent,
    ForagerComponent,
    PlantComponent,
    StigmergicPredatorComponent,
    StigmergicPreyComponent,
    stigmergic_predator_prey,
)
from ._succession import PioneerGrowthInteraction, StandComponent, SuccessionalGrowthInteraction, mangrove_succession
from ._wolf_sheep import GrassComponent, SheepComponent, WolfComponent, wolf_sheep

__all__: list[str] = [
    'AbundanceComponent',
    'CompetitionInteraction',
    'CowComponent',
    'DelayedLogisticGrowthInteraction',
    'EmotionMarksComponent',
    'ExponentialGrowthInteraction',
    'FisherKPPComponent',
    'ForagerComponent',
    'GeneralizedLotkaVolterraInteraction',
    'GrassComponent',
    'GrazingInteraction',
    'HerdingInteraction',
    'HoppingInteraction',
    'InfluenceComponent',
    'InfluenceDrawInteraction',
    'LinearGrowthInteraction',
    'LogisticGrowthInteraction',
    'MicrobeComponent',
    'MortalityInteraction',
    'MutualismInteraction',
    'NeutralReplacementInteraction',
    'NeutralSpeciesComponent',
    'OrganismComponent',
    'ParasitismInteraction',
    'PastureComponent',
    'PatchComponent',
    'PioneerGrowthInteraction',
    'PlantComponent',
    'PopulationComponent',
    'PredationInteraction',
    'PredatorComponent',
    'PredatorPreyEventsInteraction',
    'PreyComponent',
    'QuotaPhytoplanktonComponent',
    'RabbitsComponent',
    'RosenzweigMacArthurInteraction',
    'SeasonalLogisticGrowthInteraction',
    'SheepComponent',
    'StandComponent',
    'StigmergicPredatorComponent',
    'StigmergicPreyComponent',
    'SuccessionalGrowthInteraction',
    'UnstablePopulationsComponent',
    'WolfComponent',
    'agent_based_lotka_volterra',
    'competition',
    'droop_cell_quota',
    'exponential_growth',
    'fibonacci_rabbits',
    'fisher_kpp',
    'food_webs',
    'generalized_lotka_volterra',
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
