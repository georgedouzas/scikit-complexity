"""Turn kernels into the numeric functions of a form, with one registered backend per code generator."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import importlib
import warnings
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, cast

import numpy as np
from sklearn.base import BaseEstimator

from ..base import BackendError, BackendWarning, Expression, Form, Registry, Supports, ValidationError
from ..laws import Boundary, Jump, Law, Noise, Rate, Update
from ._arrays import Indices, Vector
from ._kernels import Kernel, Reference, Source, add_into, assign_into
from ._printing import SCALAR_DIALECT, print_expressions
from ._problem import Accelerated, Draws, JumpChannels, VectorFunction, WienerBlock
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
    jumps: tuple[JumpChannels, ...] = ()
    update: VectorFunction | None = None
    boundary: VectorFunction | None = None
    accelerated: Accelerated | None = None
    backends: dict[str, str] = field(default_factory=dict)


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


def _unprinted_reduction(reductions: ReductionPass) -> str | None:
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


def report_fallbacks(
    name: str,
    used: Mapping[type, list[Kernel]],
    compiled: Mapping[type, bool],
    reductions: ReductionPass,
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
    fallen = [str(cast('type[Law]', kind).law_kind) for kind in kinds if not compiled[kind]]
    if fallen:
        unprinted = _unprinted_reduction(reductions)
        reason = (
            f'{unprinted} does not print in its dialect'
            if unprinted is not None
            else 'a law holds a Python callable or reads values it cannot compile'
        )
        message = f'The {name} backend runs the {", ".join(fallen)} laws in Python, because {reason}.'
        warnings.warn(message, BackendWarning, stacklevel=3)
    return {str(cast('type[Law]', kind).law_kind): name if compiled[kind] else 'python' for kind in kinds}


BACKENDS = Registry[type[Backend]]('backend')


def _load_backends() -> None:
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
    _load_backends()
    name = ('numba' if members >= AUTO_MEMBERS else 'numpy') if backend == 'auto' else backend
    try:
        return BACKENDS.resolve(name)()
    except KeyError as error:
        names = ', '.join(['auto', *map(str, BACKENDS)])
        error_msg = f'No backend named {backend!r}. Available backends are: {names}.'
        raise BackendError(error_msg) from error
