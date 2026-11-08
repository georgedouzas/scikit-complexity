"""Forecasting realizations shared by the economics packages.

They are the forecasting heuristics.
"""

from ._heuristics import heuristics

__all__: list[str] = [
    'heuristics',
]
