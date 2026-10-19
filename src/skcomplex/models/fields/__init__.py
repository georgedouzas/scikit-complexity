"""Fields: diffusion, reaction-diffusion patterns, excitable media, and agents that read and mark fields."""

from ._foraging import Ant, Pheromone, pheromone_foraging
from ._patterns import (
    Brusselator,
    FisherKPP,
    FitzHughNagumo,
    GrayScott,
    Heat,
    Schnakenberg,
    brusselator,
    fisher_kpp,
    fitzhugh_nagumo,
    gray_scott,
    heat_diffusion,
    schnakenberg,
)

__all__: list[str] = [
    'Ant',
    'Brusselator',
    'FisherKPP',
    'FitzHughNagumo',
    'GrayScott',
    'Heat',
    'Pheromone',
    'Schnakenberg',
    'brusselator',
    'fisher_kpp',
    'fitzhugh_nagumo',
    'gray_scott',
    'heat_diffusion',
    'pheromone_foraging',
    'schnakenberg',
]
