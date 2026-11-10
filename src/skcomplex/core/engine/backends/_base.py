"""Define the base of the backends."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ...base import BaseConfig
from ...errors import ValidationError
from ...specs import LoweredFunctionsSpec, RunStateSpec, SupportSpec, Tags
from ...types import Array, Indices, VectorFunction
from ..problems._law_kernel import LawKernel
from ..problems._lowered_kernels_spec import LoweredKernelsSpec
from ..problems._output_spec import OutputSpec
from ..problems._reference_spec import ReferenceSpec
from ..types import OptionalPairs, ReadKind

AUTO_MEMBERS = 1000


class BaseBackend(BaseConfig[None, LoweredKernelsSpec]):
    """An abstract backend, which writes the numeric functions of a compiled problem.

    A backend turns the kernels of a problem into the numeric functions of its form. A backend stores its settings in
    its constructor, `prepare()` takes nothing, and `run(kernels)` takes the `LoweredKernelsSpec` of a problem, with its
    form, the size of its state vector, its reductions, and its draws, and keeps the `LoweredFunctionsSpec` of the form
    in `functions_`. Subclasses implement the hooks `_run`, `_get_name`, the name a problem records, and `_get_support`,
    the times, spaces, laws, noise, and orders the backend writes functions of.

    Attributes:
        functions_:
            The numeric functions of the form of the kernels of the last run.
    """

    functions_: LoweredFunctionsSpec

    def _prepare(self, context: None) -> None:
        """Derive nothing from the context.

        A backend prepares nothing. A backend that derives facts sets them.
        """

    def _run(self, inputs: LoweredKernelsSpec | None) -> None:
        """Set the numeric functions of the form of the kernels.

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
        """Return what the backend writes functions of.

        It is the times, spaces, laws, noise, and orders.

        Returns:
            support:
                What the backend writes functions of.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

    @classmethod
    def get_supported_tags(cls) -> Tags:
        """Return the capability tags the backend supports.

        They come from the classes of time, space, and laws it supports.

        Returns:
            tags:
                The supported times, spaces, and law kinds, where an empty field supports any value.

        Examples:
            >>> from skcomplex.core.engine.backends import JaxBackend
            >>> sorted(JaxBackend.get_supported_tags().space)
            ['none']
        """
        return cls._get_support().get_tags()

    @staticmethod
    def _get_outputs_of(kernel: LawKernel, state: RunStateSpec) -> tuple[OptionalPairs, list[Array]]:
        """Return the pairs and the values of every output of a law kernel at a state.

        Args:
            kernel:
                The law kernel.

            state:
                The state.

        Returns:
            outputs:
                The pairs in pair mode, or `None`, and one array of values per output.
        """
        _, pairs, values = kernel.run(state).values_
        return pairs, values

    @staticmethod
    def _get_adding_function(kernels: list[LawKernel], size: int) -> VectorFunction:
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
                pairs, values = BaseBackend._get_outputs_of(kernel, RunStateSpec(t, x, p))
                for item, value in zip(kernel.outputs, values, strict=True):
                    BaseBackend._add_into(output, item.target, value, pairs, p)
            return output

        return evaluate

    @staticmethod
    def _get_assigning_function(kernels: list[LawKernel]) -> VectorFunction:
        """Return the function that sets the outputs of kernels on a copy of the state.

        The state is the state vector. An output that combines with an earlier output of its target adds its change from
        the state to the value set before it, and an output that reduces over rows adds the share of every row to the
        member it writes.

        Args:
            kernels:
                The kernels.

        Returns:
            function:
                The function `f(t, x, p)`.
        """

        def evaluate(t: float, x: Array, p: Array) -> Array:
            """Set every kernel's outputs on a copy of the state.

            The changes of combined outputs are added.
            """
            output = x.copy()
            for kernel in kernels:
                _, values = BaseBackend._get_outputs_of(kernel, RunStateSpec(t, x, p))
                for item, value in zip(kernel.outputs, values, strict=True):
                    if item.reduce is not None:
                        BaseBackend._reduce_into(output, x, item, value, p)
                    elif item.combined:
                        BaseBackend._assign_change_into(output, x, item.target, value, p)
                    else:
                        BaseBackend._assign_into(output, item.target, value, p)
            return output

        return evaluate

    @staticmethod
    def _get_concatenating_function(kernels: list[LawKernel]) -> VectorFunction:
        """Return the function that concatenates the outputs of kernels.

        It gives one value per evaluation index.

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
            values = [value for kernel in kernels for value in BaseBackend._get_outputs_of(kernel, state)[1]]
            return np.concatenate([np.zeros(0), *values])

        return evaluate

    @staticmethod
    def _add_into(
        output: Array,
        target: ReferenceSpec | None,
        values: Array,
        pairs: tuple[Indices, Indices] | None,
        p: Array,
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

            p:
                The parameter vector, which holds the members of gather tables that change during a run.

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
        elif target.source is ReadKind.NEIGHBOR:
            np.add.at(column, target.shifted.get_members(p), values)
        else:
            assert pairs is not None
            np.add.at(column, pairs[0] if target.source is ReadKind.FIRST else pairs[1], values)

    @staticmethod
    def _assign_into(output: Array, target: ReferenceSpec | None, values: Array, p: Array) -> None:
        """Set the target component of a flat vector to the values of one output.

        Args:
            output:
                The flat vector to write to.

            target:
                The target component, read per member or as a single element.

            values:
                One value per evaluation index.

            p:
                The parameter vector, which holds the members of gather tables that change during a run.

        Raises:
            ValidationError:
                The output has no target.
        """
        if target is None:
            error_msg = 'An output assigned into a vector needs a target component.'
            raise ValidationError(error_msg)
        if target.source is ReadKind.NEIGHBOR:
            output[target.members][target.shifted.get_members(p)] = values
            return
        if target.source is not ReadKind.MEMBER:
            output[target.index] = values[0]
            return
        output[target.members] = values

    @staticmethod
    def _assign_change_into(output: Array, x: Array, target: ReferenceSpec | None, values: Array, p: Array) -> None:
        """Add the change of one output from the state to its target in a flat vector.

        The change goes to the target component of the vector.

        Args:
            output:
                The flat vector to write to.

            x:
                The state the output changes.

            target:
                The target component, read per member or as a single element.

            values:
                One value per evaluation index.

            p:
                The parameter vector, which holds the members of gather tables that change during a run.

        Raises:
            ValidationError:
                The output has no target.
        """
        if target is None:
            error_msg = 'An output assigned into a vector needs a target component.'
            raise ValidationError(error_msg)
        if target.source is ReadKind.NEIGHBOR:
            indices = target.shifted.get_members(p)
            np.add.at(output[target.members], indices, values - x[target.members][indices])
            return
        if target.source is not ReadKind.MEMBER:
            output[target.index] += values[0] - x[target.index]
            return
        output[target.members] += values - x[target.members]

    @staticmethod
    def _reduce_into(output: Array, x: Array, item: OutputSpec, values: Array, p: Array) -> None:
        """Add the share of every row of an output that reduces over rows to its member.

        A member written by `m` rows over every kernel takes `x + sum_r (v_r - x / m)`, the sum of its rows, or
        `x + sum_r (v_r - x) / m`, their mean, and a member written by no row keeps its value.

        Args:
            output:
                The flat vector to write to, a copy of the state.

            x:
                The state.

            item:
                The output, with its target read through a gather table and the number of rows that write the member
                of every row.

            values:
                One value per row.

            p:
                The parameter vector, which holds the members of tables that change during a run.

        Raises:
            ValidationError:
                The output has no target read through a gather table, or no counts.
        """
        target, counts = item.target, item.counts
        if target is None or target.source is not ReadKind.NEIGHBOR or counts is None:
            error_msg = 'An output that reduces over rows needs a target read through a gather table and its counts.'
            raise ValidationError(error_msg)
        indices = target.shifted.get_members(p)
        state = x[target.members][indices]
        shares = values - state / counts if item.reduce == 'sum' else (values - state) / counts
        np.add.at(output[target.members], indices, shares)
