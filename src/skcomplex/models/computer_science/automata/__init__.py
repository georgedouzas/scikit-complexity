"""Automata: elementary, life-like, noisy, and group automata, turmites, linear automata, and subshifts."""

from ._automata import (
    AntCellElement,
    ElementaryCellElement,
    LifeCellElement,
    elementary_automaton,
    game_of_life,
    langtons_ant,
)
from ._automaton import Automaton
from ._codes import Wolfram, life
from ._group_automata import (
    GoldenMeanCellElement,
    LaplacianCellElement,
    MajorityCellElement,
    ParityCellElement,
    baumslag_solitar_automaton,
    free_group_automaton,
    grigorchuk_automaton,
    heisenberg_automaton,
    lamplighter_automaton,
    linear_ring_automaton,
    subshifts,
)
from ._groups import baumslag_solitar, free_group, grigorchuk_group, heisenberg, lamplighter
from ._linear_automaton import LinearAutomaton
from ._subshift import Subshift
from ._variants import (
    DiffusionCellElement,
    NoisyElementaryCellElement,
    PascalCellElement,
    diffusion_automaton,
    highlife,
    noisy_automata,
    unbounded_alphabet_automaton,
)

__all__: list[str] = [
    'AntCellElement',
    'Automaton',
    'DiffusionCellElement',
    'ElementaryCellElement',
    'GoldenMeanCellElement',
    'LaplacianCellElement',
    'LifeCellElement',
    'LinearAutomaton',
    'MajorityCellElement',
    'NoisyElementaryCellElement',
    'ParityCellElement',
    'PascalCellElement',
    'Subshift',
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
