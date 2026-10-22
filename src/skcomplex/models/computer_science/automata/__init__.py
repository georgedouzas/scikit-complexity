"""Automata: elementary, life-like, noisy, and diffusive rules, turmites, an unbounded alphabet, and group automata."""

from ._automata import AntCell, ElementaryCell, LifeCell, elementary_automaton, game_of_life, langtons_ant
from ._codes import Wolfram, life
from ._group_automata import (
    GoldenMeanCell,
    LaplacianCell,
    MajorityCell,
    ParityCell,
    baumslag_solitar_automaton,
    free_group_automaton,
    grigorchuk_automaton,
    heisenberg_automaton,
    lamplighter_automaton,
    linear_ring_automaton,
    subshifts,
)
from ._groups import baumslag_solitar, free_group, grigorchuk_group, heisenberg, lamplighter
from ._variants import (
    DiffusionCell,
    NoisyElementaryCell,
    PascalCell,
    diffusion_automaton,
    highlife,
    noisy_automata,
    unbounded_alphabet_automaton,
)

__all__: list[str] = [
    'AntCell',
    'DiffusionCell',
    'ElementaryCell',
    'GoldenMeanCell',
    'LaplacianCell',
    'LifeCell',
    'MajorityCell',
    'NoisyElementaryCell',
    'ParityCell',
    'PascalCell',
    'Wolfram',
    'baumslag_solitar',
    'baumslag_solitar_automaton',
    'diffusion_automaton',
    'elementary_automaton',
    'free_group',
    'free_group_automaton',
    'game_of_life',
    'grigorchuk_automaton',
    'grigorchuk_group',
    'heisenberg',
    'heisenberg_automaton',
    'highlife',
    'lamplighter',
    'lamplighter_automaton',
    'langtons_ant',
    'life',
    'linear_ring_automaton',
    'noisy_automata',
    'subshifts',
    'unbounded_alphabet_automaton',
]
