"""Automata: elementary, life-like, noisy, and group automata, turmites, a linear ring, and the golden mean shift."""

from ._automata import AntCellElement, elementary_automaton, game_of_life, langtons_ant
from ._binary_cell import BinaryCellElement
from ._codes import Wolfram, life
from ._diffusion_cell import DiffusionCellElement
from ._golden_cell import GoldenCellElement
from ._group_automata import (
    LaplacianCellElement,
    baumslag_solitar_automaton,
    free_group_automaton,
    grigorchuk_automaton,
    heisenberg_automaton,
    lamplighter_automaton,
    linear_ring_automaton,
    subshifts,
)
from ._groups import baumslag_solitar, free_group, grigorchuk_group, heisenberg, lamplighter
from ._noisy_elementary_cell import NoisyElementaryCellElement
from ._pascal_cell import PascalCellElement
from ._variants import diffusion_automaton, highlife, noisy_automata, unbounded_alphabet_automaton

__all__: list[str] = [
    'AntCellElement',
    'BinaryCellElement',
    'DiffusionCellElement',
    'GoldenCellElement',
    'LaplacianCellElement',
    'NoisyElementaryCellElement',
    'PascalCellElement',
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
