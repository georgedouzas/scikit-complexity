"""Automata: elementary, life-like, noisy, and group automata, automata over infinite alphabets, and subshifts."""

from ._alphabets import (
    GroupCellElement,
    GroupDifferenceAutomatonSystem,
    PowerSeriesAutomatonSystem,
    PowerSeriesCellElement,
    RealCellElement,
    RealQuadraticAutomatonSystem,
    SelfIndexedShiftSystem,
    SubtractionAutomatonSystem,
)
from ._automata import AntCellElement, ElementaryAutomatonSystem, GameOfLifeSystem, LangtonsAntSystem
from ._binary_cell import BinaryCellElement
from ._codes import Wolfram, life
from ._dna_computation import DNAComputationSystem, LigationInteraction, StrandElement
from ._fredkin_replicator import FredkinReplicatorSystem
from ._free_group_automata import (
    LifeInATreeSystem,
    MullerAutomatonSystem,
    PairCellElement,
    SurjectiveLinearAutomatonSystem,
)
from ._group_automata import (
    BaumslagSolitarAutomatonSystem,
    FreeGroupAutomatonSystem,
    GoldenCellElement,
    GrigorchukAutomatonSystem,
    HeisenbergAutomatonSystem,
    LamplighterAutomatonSystem,
    LaplacianCellElement,
    LinearRingAutomatonSystem,
    SubshiftsSystem,
)
from ._groups import baumslag_solitar, free_group, grigorchuk_group, heisenberg, lamplighter
from ._lenia import LeniaSiteElement, LeniaSystem, LeniaUpdateInteraction
from ._line_automata import InvolutiveAutomatonSystem, NilpotentAutomatonSystem
from ._pascal_cell import PascalCellElement
from ._shift_spaces import (
    AddingMachineSystem,
    ContextFreeSubshiftSystem,
    DigitCellElement,
    GapSubshiftSystem,
    MorseElement,
    MorseSubshiftSystem,
    TernaryCellElement,
)
from ._small_world_life import SmallWorldLifeSystem
from ._turing_machine import TapeCellElement, TuringMachineSystem
from ._variants import (
    DiffusionAutomatonSystem,
    DiffusionCellElement,
    HighlifeSystem,
    NoisyAutomataSystem,
    NoisyElementaryCellElement,
    UnboundedAlphabetAutomatonSystem,
)

__all__: list[str] = [
    'AddingMachineSystem',
    'AntCellElement',
    'BaumslagSolitarAutomatonSystem',
    'BinaryCellElement',
    'ContextFreeSubshiftSystem',
    'DNAComputationSystem',
    'DiffusionAutomatonSystem',
    'DiffusionCellElement',
    'DigitCellElement',
    'ElementaryAutomatonSystem',
    'FredkinReplicatorSystem',
    'FreeGroupAutomatonSystem',
    'GameOfLifeSystem',
    'GapSubshiftSystem',
    'GoldenCellElement',
    'GrigorchukAutomatonSystem',
    'GroupCellElement',
    'GroupDifferenceAutomatonSystem',
    'HeisenbergAutomatonSystem',
    'HighlifeSystem',
    'InvolutiveAutomatonSystem',
    'LamplighterAutomatonSystem',
    'LangtonsAntSystem',
    'LaplacianCellElement',
    'LeniaSiteElement',
    'LeniaSystem',
    'LeniaUpdateInteraction',
    'LifeInATreeSystem',
    'LigationInteraction',
    'LinearRingAutomatonSystem',
    'MorseElement',
    'MorseSubshiftSystem',
    'MullerAutomatonSystem',
    'NilpotentAutomatonSystem',
    'NoisyAutomataSystem',
    'NoisyElementaryCellElement',
    'PairCellElement',
    'PascalCellElement',
    'PowerSeriesAutomatonSystem',
    'PowerSeriesCellElement',
    'RealCellElement',
    'RealQuadraticAutomatonSystem',
    'SelfIndexedShiftSystem',
    'SmallWorldLifeSystem',
    'StrandElement',
    'SubshiftsSystem',
    'SubtractionAutomatonSystem',
    'SurjectiveLinearAutomatonSystem',
    'TapeCellElement',
    'TernaryCellElement',
    'TuringMachineSystem',
    'UnboundedAlphabetAutomatonSystem',
    'Wolfram',
    'baumslag_solitar',
    'free_group',
    'grigorchuk_group',
    'heisenberg',
    'lamplighter',
    'life',
]
