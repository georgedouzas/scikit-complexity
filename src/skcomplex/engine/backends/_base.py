"""Define the base of the backends, which turn kernels into the numeric functions of a form."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ...core.base import BaseConfig
from ...core.types import Vector
from ..problems import State, SupportSpec
from ..problems._law_kernel import LawKernel
from ..problems._lowered_kernels import LoweredKernels
from ..problems._problem import VectorFunction
from ..problems._reference import Reference
from ._lowered_functions import LoweredFunctions

_AUTO_MEMBERS = 1000


class BaseBackend(BaseConfig):
    """Base of the backends, which turn the kernels of a problem into the numeric functions of its form.

    A backend stores its settings in its constructor, `prepare()` takes nothing, and `run(kernels)` takes the
    `LoweredKernels` of a problem, with its form, the size of its state vector, its reductions, and its draws, and
    returns the `LoweredFunctions` of the form. Subclasses implement the hooks `_run`, `_get_name`, the name a problem
    records, and `_get_support`, the times, spaces, laws, noise, and orders the backend writes functions of.
    """

    def _run(self, inputs: LoweredKernels) -> LoweredFunctions:
        """Return the numeric functions of the form of the kernels.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

    @classmethod
    def _get_name(cls) -> str:
        """Return the name of the backend that problems record.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the backend writes functions of.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

    @staticmethod
    def _adding_function(kernels: list[LawKernel], size: int) -> VectorFunction:
        """Return the function that sums the outputs of kernels into a zero vector.

        Args:
            kernels:
                The kernels.

            size:
                The size of the vector.

        Returns:
            function:
                The function `f(t, x, p)`.
        """

        def evaluate(t: float, x: Vector, p: Vector) -> Vector:
            """Sum every kernel's outputs into their targets."""
            output = np.zeros(size)
            for kernel in kernels:
                _, pairs, values = kernel.run(State(t, x, p))
                for item, value in zip(kernel.outputs, values, strict=True):
                    Reference._add_into(output, item.target, value, pairs)  # type: ignore[arg-type]
            return output

        return evaluate

    @staticmethod
    def _assigning_function(kernels: list[LawKernel]) -> VectorFunction:
        """Return the function that sets the outputs of kernels on a copy of the state vector.

        Args:
            kernels:
                The kernels.

        Returns:
            function:
                The function `f(t, x, p)`.
        """

        def evaluate(t: float, x: Vector, p: Vector) -> Vector:
            """Set every kernel's outputs on a copy of the state."""
            output = x.copy()
            for kernel in kernels:
                _, _, values = kernel.run(State(t, x, p))
                for item, value in zip(kernel.outputs, values, strict=True):
                    Reference._assign_into(output, item.target, value)  # type: ignore[arg-type]
            return output

        return evaluate

    @staticmethod
    def _concatenating_function(kernels: list[LawKernel]) -> VectorFunction:
        """Return the function that concatenates the outputs of kernels, one value per evaluation index.

        Args:
            kernels:
                The kernels.

        Returns:
            function:
                The function `f(t, x, p)`.
        """

        def evaluate(t: float, x: Vector, p: Vector) -> Vector:
            """Concatenate every kernel's outputs."""
            values = [value for kernel in kernels for value in kernel.run(State(t, x, p))[2]]
            return np.concatenate([np.zeros(0), *values])

        return evaluate
