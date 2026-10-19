"""Waves: vibrating strings, chains of springs, and quantum wave packets."""

from ._quantum import WaveFunction, schrodinger
from ._strings import DrivenAnchor, WaveString, spring_chain, standing_waves

__all__: list[str] = [
    'DrivenAnchor',
    'WaveFunction',
    'WaveString',
    'schrodinger',
    'spring_chain',
    'standing_waves',
]
