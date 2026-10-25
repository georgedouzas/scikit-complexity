"""Backends that write the numeric functions of compiled problems: NumPy and Numba.

Importing the package registers every backend and the solvers that run in them.
"""

from ._numba import NumbaBackend
from ._numpy import NumpyBackend

__all__: list[str] = [
    'NumbaBackend',
    'NumpyBackend',
]
