"""Turn kernels into the numeric functions of a form, with one registered backend per code generator."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import ClassVar

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ...core.types import Vector
from ..base import BaseEstimator
from ..problems import Form, SupportSpec
from ..problems._draws import Draws
from ..problems._kernel import Kernel
from ..problems._problem import VectorFunction
from ..problems._reduction_schedule import ReductionSchedule
from ..problems._reference import Reference
from ..problems._source import Source
from ._lowering import Lowering

_AUTO_MEMBERS = 1000
class BaseBackend(BaseEstimator):
    """Base class of backends, which turn kernels into the numeric functions of a form.

    `prepare(form, size)` fixes the form of the problem and the size of its state vector, which every lowering of the
    backend shares, and `run(kernels, changes, reductions=..., draws=...)` returns the `Lowering` of the kernels.
    Subclasses set the class variable `name` and implement the hook `_run`, reading the prepared `form_` and `size_`.

    Attributes:
        form_:
            The form of the problem.

        size_:
            The size of the state vector.
    """

    supports: ClassVar[SupportSpec]

    name: ClassVar[str]

    def _prepare(self, form: Form, size: int) -> None:
        """Store the form of the problem and the size of its state vector.

        Args:
            form:
                The form of the problem.

            size:
                The size of the state vector.
        """
        self.form_ = form
        self.size_ = size

    def _run(
        self,
        kernels: Mapping[type, list[Kernel]],
        changes: Mapping[int, list[tuple[Reference, float]]],
        *,
        reductions: ReductionSchedule | None = None,
        draws: Draws | None = None,
    ) -> Lowering:
        """Return the numeric functions of the prepared form.

        Args:
            kernels:
                The kernels of every law kind, in evaluation order.

            changes:
                The changes of every jump output, keyed by the output's identity.

            reductions:
                The reductions the kernels read, evaluated before them at every evaluation, or `None`.

            draws:
                The random values a stochastic map draws at every step into the parameter vector extended after its
                parameters, or `None` for a deterministic problem.

        Returns:
            lowering:
                The numeric functions.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError

    @staticmethod
    def _adding_function(kernels: list[Kernel], size: int) -> VectorFunction:
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
                _, pairs, values = kernel.evaluate(t, x, p)
                for item, value in zip(kernel.outputs, values, strict=True):
                    Reference._add_into(output, item.target, value, pairs)  # type: ignore[arg-type]
            return output

        return evaluate

    @staticmethod
    def _assigning_function(kernels: list[Kernel]) -> VectorFunction:
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
                _, _, values = kernel.evaluate(t, x, p)
                for item, value in zip(kernel.outputs, values, strict=True):
                    Reference._assign_into(output, item.target, value)  # type: ignore[arg-type]
            return output

        return evaluate

    @staticmethod
    def _concatenating_function(kernels: list[Kernel]) -> VectorFunction:
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
            values = [value for kernel in kernels for value in kernel.evaluate(t, x, p)[2]]
            return np.concatenate([np.zeros(0), *values])

        return evaluate

    @staticmethod
    def _reads_reduction(kernel: Kernel, item: Expression) -> bool:
        """Check whether a symbol of a kernel reads a reduction, directly or through the operand of a derived read."""
        reference = kernel.references[item]
        if reference.source is Source.REDUCED:
            return True
        if reference.source is Source.DERIVED:
            return any(BaseBackend._reads_reduction(kernel, symbol) for symbol in reference.neighbor.symbols)
        return False

    @staticmethod
    def _reads_reductions(kernels: list[Kernel]) -> bool:
        """Check whether the outputs of kernels read reductions, so their function evaluates the reductions first.

        Args:
            kernels:
                The kernels of one law kind.

        Returns:
            reads:
                Whether an output reads a reduction.
        """
        return any(BaseBackend._reads_reduction(kernel, item) for kernel in kernels for item in kernel.arguments)
