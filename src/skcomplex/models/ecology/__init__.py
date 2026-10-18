"""Population ecology: growth, mortality, and predation, as equations and as wolf and sheep agents."""

from ._factories import exponential_growth, logistic_growth, lotka_volterra
from ._populations import ExponentialGrowth, LogisticGrowth, Mortality, Population, Predation
from ._wolf_sheep import Grass, Sheep, Wolf, wolf_sheep

__all__: list[str] = [
    'ExponentialGrowth',
    'Grass',
    'LogisticGrowth',
    'Mortality',
    'Population',
    'Predation',
    'Sheep',
    'Wolf',
    'exponential_growth',
    'logistic_growth',
    'lotka_volterra',
    'wolf_sheep',
]
