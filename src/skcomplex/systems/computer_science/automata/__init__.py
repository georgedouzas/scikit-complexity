"""Automata: elementary, life-like, noisy, and group automata, automata over infinite alphabets, and subshifts."""

from ._alphabets import (
    group_difference_automaton,
    power_series_automaton,
    real_quadratic_automaton,
    self_indexed_shift,
    subtraction_automaton,
)
from ._automata import AntCellElement, elementary_automaton, game_of_life, langtons_ant
from ._binary_cell import BinaryCellElement
from ._codes import Wolfram, life
from ._diffusion_cell import DiffusionCellElement
from ._digit_cell import DigitCellElement
from ._free_group_automata import life_in_a_tree, muller_automaton, surjective_linear_automaton
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
from ._group_cell import GroupCellElement
from ._groups import baumslag_solitar, free_group, grigorchuk_group, heisenberg, lamplighter
from ._line_automata import involutive_automaton, nilpotent_automaton
from ._morse_word import MorseElement
from ._noisy_elementary_cell import NoisyElementaryCellElement
from ._pair_cell import PairCellElement
from ._pascal_cell import PascalCellElement
from ._power_series_cell import PowerSeriesCellElement
from ._real_cell import RealCellElement
from ._shift_spaces import adding_machine, context_free_subshift, gap_subshift, morse_subshift
from ._ternary_cell import TernaryCellElement
from ._variants import diffusion_automaton, highlife, noisy_automata, unbounded_alphabet_automaton

__all__: list[str] = [
    'AntCellElement',
    'BinaryCellElement',
    'DiffusionCellElement',
    'DigitCellElement',
    'GoldenCellElement',
    'GroupCellElement',
    'LaplacianCellElement',
    'MorseElement',
    'NoisyElementaryCellElement',
    'PairCellElement',
    'PascalCellElement',
    'PowerSeriesCellElement',
    'RealCellElement',
    'TernaryCellElement',
    'Wolfram',
    'adding_machine',
    'baumslag_solitar',
    'baumslag_solitar_automaton',
    'context_free_subshift',
    'diffusion_automaton',
    'elementary_automaton',
    'free_group',
    'free_group_automaton',
    'game_of_life',
    'gap_subshift',
    'grigorchuk_automaton',
    'grigorchuk_group',
    'group_difference_automaton',
    'heisenberg',
    'heisenberg_automaton',
    'highlife',
    'involutive_automaton',
    'lamplighter',
    'lamplighter_automaton',
    'langtons_ant',
    'life',
    'life_in_a_tree',
    'linear_ring_automaton',
    'morse_subshift',
    'muller_automaton',
    'nilpotent_automaton',
    'noisy_automata',
    'power_series_automaton',
    'real_quadratic_automaton',
    'self_indexed_shift',
    'subshifts',
    'subtraction_automaton',
    'surjective_linear_automaton',
    'unbounded_alphabet_automaton',
]
