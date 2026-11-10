"""Define backends, which write the numeric functions of compiled problems.

They are NumPy, the default, Numba, which compiles kernels, and JAX, which differentiates them.
"""

from ._base import BaseBackend
from ._jax import JaxBackend
from ._numba import NumbaBackend
from ._numpy import NumpyBackend

__all__: list[str] = [
    'BaseBackend',
    'JaxBackend',
    'NumbaBackend',
    'NumpyBackend',
]
