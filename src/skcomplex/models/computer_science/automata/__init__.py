"""Automata: elementary, life-like, noisy, and diffusive rules, turmites, and an unbounded alphabet."""

from ._automata import AntCell, ElementaryCell, LifeCell, elementary_automaton, game_of_life, langtons_ant
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
    'LifeCell',
    'NoisyElementaryCell',
    'PascalCell',
    'diffusion_automaton',
    'elementary_automaton',
    'game_of_life',
    'highlife',
    'langtons_ant',
    'noisy_automata',
    'unbounded_alphabet_automaton',
]
