"""Define the backends, which write the numeric functions solvers call.

They are NumPy, the default, Numba, which compiles kernels, and JAX, which differentiates them.
"""

from ._base import BaseBackend
from ._jax import JaxBackend
from ._lowered_functions_spec import LoweredFunctionsSpec
from ._numba import NumbaBackend
from ._numpy import NumpyBackend
from ._support_spec import SupportSpec

__all__: list[str] = [
    'BaseBackend',
    'JaxBackend',
    'LoweredFunctionsSpec',
    'NumbaBackend',
    'NumpyBackend',
    'SupportSpec',
]
