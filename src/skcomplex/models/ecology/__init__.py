"""Population ecology: growth, mortality, and predation."""

from ._populations import ExponentialGrowth, LogisticGrowth, Mortality, Population, Predation

__all__: list[str] = [
    'ExponentialGrowth',
    'LogisticGrowth',
    'Mortality',
    'Population',
    'Predation',
]
