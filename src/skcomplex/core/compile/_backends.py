"""Turn kernels into the numeric functions of a form, with one registered backend per code generator."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import importlib
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from sklearn.base import BaseEstimator

from ..base import BackendError, Expression, Form, Registry, Supports
from ..laws import Boundary, Jump, Noise, Rate, Update
from ._kernels import Indices, Kernel, Reference, Source, add_into, assign_into
from ._problem import Accelerated, Draws, JumpChannels, Vector, VectorFunction, WienerBlock
from ._reductions import ReductionPass

AUTO_MEMBERS = 1000
FUNCTION_KINDS: dict[str, type] = {
    'drift': Rate,
    'diffusion': Noise,
    'propensities': Jump,
    'update': Update,
    'boundary': Boundary,
}


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
    """

    drift: VectorFunction | None = None
    diffusion: VectorFunction | None = None
    noise_rows: Indices = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    noise_columns: Indices = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    n_processes: int = 0
    noise_blocks: tuple[WienerBlock, ...] = ()
    propensities: VectorFunction | None = None
    jumps: tuple[JumpChannels, ...] = ()
    update: VectorFunction | None = None
    boundary: VectorFunction | None = None
    accelerated: Accelerated | None = None


class Backend(BaseEstimator):
    """Base class of backends, which turn kernels into the numeric functions of a form.

    Subclasses set the class variable `name` and implement `lower`.
    """

    supports: ClassVar[Supports]

    name: ClassVar[str]

    @abstractmethod
    def lower(
        self,
        form: Form,
        kernels: Mapping[type, list[Kernel]],
        changes: Mapping[int, list[tuple[Reference, float]]],
        size: int,
        *,
        reductions: ReductionPass | None = None,
        draws: Draws | None = None,
    ) -> Lowering:
        """Return the numeric functions of a form.

        Args:
            form:
                The form of the problem.

            kernels:
                The kernels of every law kind, in evaluation order.

            changes:
                The changes of every jump output, keyed by the output's identity.

            size:
                The size of the state vector.

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
                The subclass does not implement the method.
        """
        raise NotImplementedError


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


def _reads_reduction(kernel: Kernel, item: Expression) -> bool:
    """Check whether a symbol of a kernel reads a reduction, directly or through the operand of a derived read."""
    reference = kernel.references[item]
    if reference.source is Source.REDUCED:
        return True
    if reference.source is Source.DERIVED:
        return any(_reads_reduction(kernel, symbol) for symbol in reference.neighbor.symbols)
    return False


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


def with_reductions(
    lowering: Lowering,
    reductions: ReductionPass | None,
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


BACKENDS = Registry[type[Backend]]('backend')


def load_backends() -> None:
    """Import `skcomplex.core.backends`, whose modules register the backends and the solvers that run in them."""
    importlib.import_module('skcomplex.core.backends')


def register_backend(backend: type[Backend]) -> None:
    """Register a backend class under its name.

    Args:
        backend:
            The backend class.
    """
    BACKENDS.register(backend.name, backend)


def resolve_backend(backend: str | Backend, members: int) -> Backend:
    """Return a backend from a registered name, `'auto'`, or an instance.

    `'auto'` picks `numba` when the groups of the system hold at least 1000 members in total, and `numpy` otherwise.

    Args:
        backend:
            A registered name, `'auto'`, or a backend instance.

        members:
            The total number of members of the system's groups.

    Returns:
        backend:
            A new backend for a name, or the given instance.

    Raises:
        BackendError:
            No backend is registered under the name.
    """
    if isinstance(backend, Backend):
        return backend
    load_backends()
    name = ('numba' if members >= AUTO_MEMBERS else 'numpy') if backend == 'auto' else backend
    try:
        return BACKENDS.resolve(name)()
    except KeyError as error:
        names = ', '.join(['auto', *map(str, BACKENDS)])
        error_msg = f'No backend named {backend!r}. Available backends are: {names}.'
        raise BackendError(error_msg) from error
