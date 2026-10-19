"""Collective behaviour: animals and people that coordinate through signals, trails, and contact."""

from ._decisions import Branch, Nervousness, Rats, TrailChoice, binary_ant_decision, rats
from ._flocks import Boid, MorsePotential, SelfPropulsion, bird_swarm, boids
from ._foraging import Ant, Pheromone, pheromone_foraging
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
    'TrailChoice',
    'Walker',
    'active_walkers',
    'binary_ant_decision',
    'bird_swarm',
    'boids',
    'pheromone_foraging',
    'rats',
]
