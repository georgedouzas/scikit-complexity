"""Analyse systems symbolically: derive their equations and solve them exactly."""

from ._symbolic import equations_of, solve_symbolically

__all__: list[str] = [
    'equations_of',
    'solve_symbolically',
]
