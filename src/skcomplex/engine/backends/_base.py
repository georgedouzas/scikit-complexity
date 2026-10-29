"""Define the base of the backends, which turn kernels into the numeric functions of a form."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ...base import BaseConfig
from ...specs import LoweredFunctionsSpec, RunStateSpec, SupportSpec
from ...types import Array, Indices, ReadKind, VectorFunction
from ..problems import LoweredKernelsSpec, ReferenceSpec
from ..problems._law_kernel import LawKernel

_AUTO_MEMBERS = 1000


class BaseBackend(BaseConfig[None, LoweredKernelsSpec]):
    """Base of the backends, which turn the kernels of a problem into the numeric functions of its form.

    A backend stores its settings in its constructor, `prepare()` takes nothing, and `run(kernels)` takes the
    `LoweredKernelsSpec` of a problem, with its form, the size of its state vector, its reductions, and its draws, and
    returns the `LoweredFunctionsSpec` of the form. Subclasses implement the hooks `_run`, `_get_name`, the name a
    problem records, and `_get_support`, the times, spaces, laws, noise, and orders the backend writes functions of.
    """

    def _run(self, inputs: LoweredKernelsSpec | None) -> LoweredFunctionsSpec:
        """Return the numeric functions of the form of the kernels.

        Args:
            inputs:
                The kernels of a problem.

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

        def evaluate(t: float, x: Array, p: Array) -> Array:
            """Sum every kernel's outputs into their targets."""
            output = np.zeros(size)
            for kernel in kernels:
                _, pairs, values = kernel.run(RunStateSpec(t, x, p))
                for item, value in zip(kernel.outputs, values, strict=True):
                    BaseBackend._add_into(output, item.target, value, pairs)  # type: ignore[arg-type]
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

        def evaluate(t: float, x: Array, p: Array) -> Array:
            """Set every kernel's outputs on a copy of the state."""
            output = x.copy()
            for kernel in kernels:
                _, _, values = kernel.run(RunStateSpec(t, x, p))
                for item, value in zip(kernel.outputs, values, strict=True):
                    BaseBackend._assign_into(output, item.target, value)  # type: ignore[arg-type]
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

        def evaluate(t: float, x: Array, p: Array) -> Array:
            """Concatenate every kernel's outputs."""
            values = [value for kernel in kernels for value in kernel.run(RunStateSpec(t, x, p))[2]]
            return np.concatenate([np.zeros(0), *values])

        return evaluate

    @staticmethod
    def _add_into(output: Array, target: ReferenceSpec, values: Array, pairs: tuple[Indices, Indices] | None) -> None:
        """Add the values of one output to the target component of a flat vector.

        Args:
            output:
                The flat vector to add to.

            target:
                The target component.

            values:
                One value per evaluation index.

            pairs:
                The pairs of a pair kernel, or `None` in member mode.
        """
        if target.source is ReadKind.SINGLE:
            output[target.index] += values.sum()
            return
        column = output[target.members]
        if target.source is ReadKind.MEMBER:
            column += values
        else:
            assert pairs is not None
            np.add.at(column, pairs[0] if target.source is ReadKind.FIRST else pairs[1], values)

    @staticmethod
    def _assign_into(output: Array, target: ReferenceSpec, values: Array) -> None:
        """Set the target component of a flat vector to the values of one output.

        Args:
            output:
                The flat vector to write to.

            target:
                The target component, read per member or as a single element.

            values:
                One value per evaluation index.
        """
        if target.source is not ReadKind.MEMBER:
            output[target.index] = values[0]
            return
        output[target.members] = values
