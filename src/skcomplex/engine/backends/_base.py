"""Turn kernels into the numeric functions of a form, with one registered backend per code generator."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, cast

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ...core.errors import BackendWarning, ValidationError
from ...core.laws import BoundaryLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ...core.types import Indices, Vector
from ..problems import Form, SupportSpec
from ..problems._kernels import Kernel, Reference, Source, add_into, assign_into
from ..problems._printing import SCALAR_DIALECT, print_expressions
from ..problems._problem import Accelerated, Draws, VectorFunction, WienerBlock, _JumpChannels
from ..problems._reductions import ReductionSchedule
from ..utils import BaseEstimator

AUTO_MEMBERS = 1000
FUNCTION_KINDS: dict[str, type] = {
    'drift': RateLaw,
    'diffusion': NoiseLaw,
    'propensities': JumpLaw,
    'update': UpdateLaw,
    'boundary': BoundaryLaw,
}


def _reads_reduction(kernel: Kernel, item: Expression) -> bool:
    """Check whether a symbol of a kernel reads a reduction, directly or through the operand of a derived read."""
    reference = kernel.references[item]
    if reference.source is Source.REDUCED:
        return True
    if reference.source is Source.DERIVED:
        return any(_reads_reduction(kernel, symbol) for symbol in reference.neighbor.symbols)
    return False


def _unprinted_reduction(reductions: ReductionSchedule) -> str | None:
    """Return the first reduction that does not print in the scalar dialect, or `None` when every one prints."""
    for reduction in reductions.reductions:
        try:
            print_expressions(
                [reduction.argument, reduction.condition if reduction.condition is not None else 1],
                SCALAR_DIALECT,
            )
        except ValidationError:
            return f'the {reduction.kind} reduction {reduction.argument}'
    for neighbourhood in reductions.neighbourhoods:
        try:
            print_expressions([neighbourhood.argument, neighbourhood.weight], SCALAR_DIALECT)
        except ValidationError:
            return f'the neighbourhood {neighbourhood.kind} reduction {neighbourhood.argument}'
    return None


def adding_function(kernels: list[Kernel], size: int) -> VectorFunction:
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
                add_into(output, item.target, value, pairs)  # type: ignore[arg-type]
        return output

    return evaluate


def assigning_function(kernels: list[Kernel]) -> VectorFunction:
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
                assign_into(output, item.target, value)  # type: ignore[arg-type]
        return output

    return evaluate


def concatenating_function(kernels: list[Kernel]) -> VectorFunction:
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


def reads_reductions(kernels: list[Kernel]) -> bool:
    """Check whether the outputs of kernels read reductions, so their function evaluates the reductions first.

    Args:
        kernels:
            The kernels of one law kind.

    Returns:
        reads:
            Whether an output reads a reduction.
    """
    return any(_reads_reduction(kernel, item) for kernel in kernels for item in kernel.arguments)


@dataclass
class Lowering:
    """Numeric functions of a form produced by a backend.

    Args:
        drift:
            The drift function, or `None`.

        diffusion:
            The function that returns every Wiener process amplitude, or `None`.

        noise_rows:
            The state index each noise amplitude drives.

        noise_columns:
            The Wiener process each noise amplitude draws.

        n_processes:
            The number of Wiener processes.

        noise_blocks:
            The blocks of correlated processes with their covariance factors.

        propensities:
            The function that returns every jump channel rate, or `None`.

        jumps:
            The channels of every jump law.

        update:
            The next-state function of a discrete map, or `None`.

        boundary:
            The function that applies the boundary laws, or `None`.

        accelerated:
            The compiled stepper loops, or `None` when the backend has none.

        backends:
            The backend that runs the kernels of every law kind, by the kind's name, when it differs by kind.
    """

    drift: VectorFunction | None = None
    diffusion: VectorFunction | None = None
    noise_rows: Indices = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    noise_columns: Indices = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    n_processes: int = 0
    noise_blocks: tuple[WienerBlock, ...] = ()
    propensities: VectorFunction | None = None
    jumps: tuple[_JumpChannels, ...] = ()
    update: VectorFunction | None = None
    boundary: VectorFunction | None = None
    accelerated: Accelerated | None = None
    backends: dict[str, str] = field(default_factory=dict)


def with_reductions(
    lowering: Lowering,
    reductions: ReductionSchedule | None,
    kernels: Mapping[type, list[Kernel]] | None = None,
) -> Lowering:
    """Return a lowering whose functions evaluate the reductions at the state they receive before their kernels.

    Args:
        lowering:
            The functions of the extended parameter vector.

        reductions:
            The reductions, or `None`.

        kernels:
            The kernels of every law kind, so that only the functions whose kernels read reductions evaluate them,
            or `None` for every function.

    Returns:
        lowering:
            The functions of the parameter vector.
    """
    if reductions is None or not reductions.steps:
        return lowering
    for name, kind in FUNCTION_KINDS.items():
        if kernels is None or reads_reductions(kernels.get(kind, [])):
            setattr(lowering, name, reductions.wrap(getattr(lowering, name)))
    return lowering


def report_fallbacks(
    name: str,
    used: Mapping[type, list[Kernel]],
    compiled: Mapping[type, bool],
    reductions: ReductionSchedule,
) -> dict[str, str]:
    """Warn once about the law kinds whose kernels a compiling backend runs in Python, and return every kind's runner.

    Args:
        name:
            The name of the compiling backend.

        used:
            The kernels of every law kind the backend lowers.

        compiled:
            Whether the backend compiles the kernels of every law kind.

        reductions:
            The reductions the kernels read.

    Returns:
        backends:
            The backend name for every law kind with kernels, `'python'` for those that run in Python.
    """
    kinds = [kind for kind, items in used.items() if items]
    fallen = [cast('type', kind).__name__ for kind in kinds if not compiled[kind]]
    if fallen:
        unprinted = _unprinted_reduction(reductions)
        reason = (
            f'{unprinted} does not print in its dialect'
            if unprinted is not None
            else 'a law holds a Python callable or reads values it cannot compile'
        )
        message = f'The {name} backend runs the {", ".join(fallen)} laws in Python, because {reason}.'
        warnings.warn(message, BackendWarning, stacklevel=4)
    return {cast('type', kind).__name__: name if compiled[kind] else 'python' for kind in kinds}


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
