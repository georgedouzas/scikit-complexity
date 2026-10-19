"""Automata: elementary rules, life, and turmites."""

from ._automata import AntCell, ElementaryCell, LifeCell, elementary_automaton, game_of_life, langtons_ant

__all__: list[str] = [
    'AntCell',
    'ElementaryCell',
    'LifeCell',
    'elementary_automaton',
    'game_of_life',
    'langtons_ant',
]
