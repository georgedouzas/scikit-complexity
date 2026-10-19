"""Ecology: population growth, predation, and invasion fronts."""

from ._factories import Rabbits, exponential_growth, fibonacci_rabbits, hutchinson, logistic_growth, lotka_volterra
from ._fronts import FisherKPP, fisher_kpp
from ._populations import DelayedLogisticGrowth, ExponentialGrowth, LogisticGrowth, Mortality, Population, Predation
from ._wolf_sheep import Grass, Sheep, Wolf, wolf_sheep

__all__: list[str] = [
    'DelayedLogisticGrowth',
    'ExponentialGrowth',
    'FisherKPP',
    'Grass',
    'LogisticGrowth',
    'Mortality',
    'Population',
    'Predation',
    'Rabbits',
    'Sheep',
    'Wolf',
    'exponential_growth',
    'fibonacci_rabbits',
    'fisher_kpp',
    'hutchinson',
    'logistic_growth',
    'lotka_volterra',
    'wolf_sheep',
]
