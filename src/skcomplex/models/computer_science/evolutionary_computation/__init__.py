"""Evolutionary computation: genetic algorithms and bit strings on fitness landscapes."""

from ._genetic_algorithms import Evolution, Strategy, robby_ga
from ._landscapes import BitString, Reproduction, bit_string_evolution

__all__: list[str] = [
    'BitString',
    'Evolution',
    'Reproduction',
    'Strategy',
    'bit_string_evolution',
    'robby_ga',
]
