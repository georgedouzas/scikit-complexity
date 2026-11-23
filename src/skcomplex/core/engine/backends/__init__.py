"""Define the backends, which write the numeric functions that solvers call.

They are NumPy, the default, Numba, which compiles the functions to machine code for large groups, and JAX, which also
gives their derivatives, such as `LogisticGrowthSystem(backend=NumbaBackend())`.
"""

from ._base import BaseBackend
from ._jax import JaxBackend
from ._numba import NumbaBackend
from ._numpy import NumpyBackend
from ._support_spec import SupportSpec

__all__: list[str] = [
    'BaseBackend',
    'JaxBackend',
    'NumbaBackend',
    'NumpyBackend',
    'SupportSpec',
]
