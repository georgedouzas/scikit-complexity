"""Ecology: population growth, predation, competition, food webs, succession, grazing, and neutral communities."""

from ._agent_predation import Predator, Prey, agent_based_lotka_volterra
from ._cell_quota import QuotaPhytoplankton, droop_cell_quota
from ._clustering import Hopping, Patch, neutral_clustering
from ._communities import Competition, Mutualism, competition, food_webs, mutualism
from ._factories import (
    Rabbits,
    exponential_growth,
    fibonacci_rabbits,
    hutchinson,
    logistic_growth,
    lotka_volterra,
    varying_capacity_logistic,
)
from ._fronts import FisherKPP, fisher_kpp
from ._meadow import Cow, Grazing, Herding, Pasture, meadow_capacity
from ._microbial_communities import GeneralizedLotkaVolterra, Microbe, generalized_lotka_volterra
from ._neutral import NeutralSpecies, neutral_macroecology
from ._populations import (
    DelayedLogisticGrowth,
    ExponentialGrowth,
    LogisticGrowth,
    Mortality,
    Population,
    Predation,
    SeasonalLogisticGrowth,
)
from ._predation import (
    Parasitism,
    PredatorPreyEvents,
    RosenzweigMacArthur,
    lotka_volterra_with_harvesting,
    nicholson_bailey,
    rosenzweig_macarthur,
    stochastic_predator_prey,
)
from ._speciation import NeutralReplacement, Organism, neutral_biodiversity
from ._stigmergy import (
    EmotionMarks,
    Forager,
    Plant,
    StigmergicPredator,
    StigmergicPrey,
    stigmergic_predator_prey,
)
from ._succession import PioneerGrowth, Stand, SuccessionalGrowth, mangrove_succession
from ._wolf_sheep import Grass, Sheep, Wolf, wolf_sheep

__all__: list[str] = [
    'Competition',
    'Cow',
    'DelayedLogisticGrowth',
    'EmotionMarks',
    'ExponentialGrowth',
    'FisherKPP',
    'Forager',
    'GeneralizedLotkaVolterra',
    'Grass',
    'Grazing',
    'Herding',
    'Hopping',
    'LogisticGrowth',
    'Microbe',
    'Mortality',
    'Mutualism',
    'NeutralReplacement',
    'NeutralSpecies',
    'Organism',
    'Parasitism',
    'Pasture',
    'Patch',
    'PioneerGrowth',
    'Plant',
    'Population',
    'Predation',
    'Predator',
    'PredatorPreyEvents',
    'Prey',
    'QuotaPhytoplankton',
    'Rabbits',
    'RosenzweigMacArthur',
    'SeasonalLogisticGrowth',
    'Sheep',
    'Stand',
    'StigmergicPredator',
    'StigmergicPrey',
    'SuccessionalGrowth',
    'Wolf',
    'agent_based_lotka_volterra',
    'competition',
    'droop_cell_quota',
    'exponential_growth',
    'fibonacci_rabbits',
    'fisher_kpp',
    'food_webs',
    'generalized_lotka_volterra',
    'hutchinson',
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
    'stigmergic_predator_prey',
    'stochastic_predator_prey',
    'varying_capacity_logistic',
    'wolf_sheep',
]
