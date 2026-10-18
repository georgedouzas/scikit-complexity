"""Social systems: Schelling segregation, Boltzmann wealth exchange, and the voter model."""

from ._agents import Resident, Trader, Voter, boltzmann_wealth, schelling, voter_model

__all__: list[str] = [
    'Resident',
    'Trader',
    'Voter',
    'boltzmann_wealth',
    'schelling',
    'voter_model',
]
