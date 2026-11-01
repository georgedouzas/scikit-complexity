"""Evolutionary computation: genetic algorithms and bit strings on fitness landscapes."""

from ._bit_string import BitStringElement
from ._genetic_algorithms import robby_ga
from ._landscapes import bit_string_evolution
from ._reproduction import ReproductionInteraction
from ._room_cleaning import RoomCleaningInteraction
from ._strategy import StrategyElement

__all__: list[str] = [
    'BitStringElement',
    'ReproductionInteraction',
    'RoomCleaningInteraction',
    'StrategyElement',
    'bit_string_evolution',
    'robby_ga',
]
