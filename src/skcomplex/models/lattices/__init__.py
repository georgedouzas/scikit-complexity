"""Lattice models: cellular automata, self-organised criticality, traffic, and Monte Carlo spin systems."""

from ._automata import (
    AntCell,
    ElementaryCell,
    HodgepodgeCell,
    LifeCell,
    Road,
    Voter,
    elementary_automaton,
    game_of_life,
    hodgepodge,
    langtons_ant,
    majority_rule,
    nagel_schreckenberg,
)
from ._criticality import Fault, Forest, SandpileCell, earthquakes, forest_fire, sandpile
from ._spins import IsingSpin, PottsSpin, XYSpin, ising, potts, xy

__all__: list[str] = [
    'AntCell',
    'ElementaryCell',
    'Fault',
    'Forest',
    'HodgepodgeCell',
    'IsingSpin',
    'LifeCell',
    'PottsSpin',
    'Road',
    'SandpileCell',
    'Voter',
    'XYSpin',
    'earthquakes',
    'elementary_automaton',
    'forest_fire',
    'game_of_life',
    'hodgepodge',
    'ising',
    'langtons_ant',
    'majority_rule',
    'nagel_schreckenberg',
    'potts',
    'sandpile',
    'xy',
]
