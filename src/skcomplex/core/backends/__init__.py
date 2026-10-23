"""Backends that run compiled problems: NumPy, Numba, and Mesa for agents.

Importing the package registers every backend and the solvers that run in them.
"""

from ._base import ExternalBackend
from ._mesa import AgentView, MesaStepper, ModelView
from ._numba import NumbaBackend
from ._numpy import NumpyBackend

__all__: list[str] = [
    'AgentView',
    'ExternalBackend',
    'MesaStepper',
    'ModelView',
    'NumbaBackend',
    'NumpyBackend',
]
