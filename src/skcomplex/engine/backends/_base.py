"""Define the base of the backends, which turn kernels into the numeric functions of a form."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ...base import BaseConfig
from ...errors import ValidationError
from ...specs import LoweredFunctionsSpec, RunStateSpec, SupportSpec
from ...types import Array, Indices, VectorFunction
from ..problems import LoweredKernelsSpec, ReferenceSpec
from ..problems._law_kernel import LawKernel
from ..types import OptionalPairs, ReadKind

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

        Returns:
            functions:
                The numeric functions of the form.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

    @classmethod
    def _get_name(cls) -> str:
        """Return the name of the backend that problems record.

        Returns:
            name:
                The name of the backend.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the backend writes functions of.

        Returns:
            support:
                What the backend writes functions of.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

    @staticmethod
    def _outputs_of(kernel: LawKernel, state: RunStateSpec) -> tuple[OptionalPairs, list[Array]]:
        """Return the pairs and the values of every output of a law kernel at a state.

        Args:
            kernel:
                The law kernel.

            state:
                The state.

        Returns:
            outputs:
                The pairs in pair mode, or `None`, and one array of values per output.

        Raises:
            TypeError:
                The kernel does not return its number of indices, its pairs, and its values.
        """
        result = kernel.run(state)
        if not isinstance(result, tuple):
            error_msg = (
                f'A law kernel returns its indices, pairs, and values as a tuple, not a {type(result).__name__}.'
            )
            raise TypeError(error_msg)
        _, pairs, values = result
        return pairs, values

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
                pairs, values = BaseBackend._outputs_of(kernel, RunStateSpec(t, x, p))
                for item, value in zip(kernel.outputs, values, strict=True):
                    BaseBackend._add_into(output, item.target, value, pairs)
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
                _, values = BaseBackend._outputs_of(kernel, RunStateSpec(t, x, p))
                for item, value in zip(kernel.outputs, values, strict=True):
                    BaseBackend._assign_into(output, item.target, value)
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
            state = RunStateSpec(t, x, p)
            values = [value for kernel in kernels for value in BaseBackend._outputs_of(kernel, state)[1]]
            return np.concatenate([np.zeros(0), *values])

        return evaluate

    @staticmethod
    def _add_into(
        output: Array,
        target: ReferenceSpec | None,
        values: Array,
        pairs: tuple[Indices, Indices] | None,
    ) -> None:
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

        Raises:
            ValidationError:
                The output has no target.
        """
        if target is None:
            error_msg = 'An output added into a vector needs a target component.'
            raise ValidationError(error_msg)
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
    def _assign_into(output: Array, target: ReferenceSpec | None, values: Array) -> None:
        """Set the target component of a flat vector to the values of one output.

        Args:
            output:
                The flat vector to write to.

            target:
                The target component, read per member or as a single element.

            values:
                One value per evaluation index.

        Raises:
            ValidationError:
                The output has no target.
        """
        if target is None:
            error_msg = 'An output assigned into a vector needs a target component.'
            raise ValidationError(error_msg)
        if target.source is not ReadKind.MEMBER:
            output[target.index] = values[0]
            return
        output[target.members] = values
