"""Automata: elementary, life-like, noisy, and group automata, automata over infinite alphabets, and subshifts."""

from ._alphabets import (
    GroupCellElement,
    PowerSeriesCellElement,
    RealCellElement,
    group_difference_automaton,
    power_series_automaton,
    real_quadratic_automaton,
    self_indexed_shift,
    subtraction_automaton,
)
from ._automata import AntCellElement, elementary_automaton, game_of_life, langtons_ant
from ._binary_cell import BinaryCellElement
from ._codes import Wolfram, life
from ._dna_computation import LigationInteraction, StrandElement, dna_computation
from ._fredkin_replicator import fredkin_replicator
from ._free_group_automata import PairCellElement, life_in_a_tree, muller_automaton, surjective_linear_automaton
from ._group_automata import (
    GoldenCellElement,
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
from ._lenia import LeniaSiteElement, LeniaUpdateInteraction, lenia
from ._line_automata import involutive_automaton, nilpotent_automaton
from ._pascal_cell import PascalCellElement
from ._shift_spaces import (
    DigitCellElement,
    MorseElement,
    TernaryCellElement,
    adding_machine,
    context_free_subshift,
    gap_subshift,
    morse_subshift,
)
from ._small_world_life import small_world_life
from ._turing_machine import TapeCellElement, turing_machine
from ._variants import (
    DiffusionCellElement,
    NoisyElementaryCellElement,
    diffusion_automaton,
    highlife,
    noisy_automata,
    unbounded_alphabet_automaton,
)

__all__: list[str] = [
    'AntCellElement',
    'BinaryCellElement',
    'DiffusionCellElement',
    'DigitCellElement',
    'GoldenCellElement',
    'GroupCellElement',
    'LaplacianCellElement',
    'LeniaSiteElement',
    'LeniaUpdateInteraction',
    'LigationInteraction',
    'MorseElement',
    'NoisyElementaryCellElement',
    'PairCellElement',
    'PascalCellElement',
    'PowerSeriesCellElement',
    'RealCellElement',
    'StrandElement',
    'TapeCellElement',
    'TernaryCellElement',
    'Wolfram',
    'adding_machine',
    'baumslag_solitar',
    'baumslag_solitar_automaton',
    'context_free_subshift',
    'diffusion_automaton',
    'dna_computation',
    'elementary_automaton',
    'fredkin_replicator',
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
    'lenia',
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
    'small_world_life',
    'subshifts',
    'subtraction_automaton',
    'surjective_linear_automaton',
    'turing_machine',
    'unbounded_alphabet_automaton',
]
