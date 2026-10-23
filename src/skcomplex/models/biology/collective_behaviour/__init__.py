"""Collective behaviour: animals and people that coordinate through signals, trails, and contact."""

from ._decisions import Branch, Nervousness, Rats, TrailChoice, binary_ant_decision, rats
from ._flocks import Boid, MorsePotential, SelfPropulsion, bird_swarm, boids
from ._foraging import Ant, Pheromone, pheromone_foraging
from ._leadership import ShoalMember, couzin_leadership
from ._trails import Ground, Walker, active_walkers

__all__: list[str] = [
    'Ant',
    'Boid',
    'Branch',
    'Ground',
    'MorsePotential',
    'Nervousness',
    'Pheromone',
    'Rats',
    'SelfPropulsion',
    'ShoalMember',
    'TrailChoice',
    'Walker',
    'active_walkers',
    'binary_ant_decision',
    'bird_swarm',
    'boids',
    'couzin_leadership',
    'pheromone_foraging',
    'rats',
]
