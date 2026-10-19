"""Turn kernels into the numeric functions of a form, with one registered backend per code generator."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from sklearn.base import BaseEstimator

from ..base import BackendError, Form, Registry
from ..model import Boundary, Jump, Noise, Rate, Update
from ._kernels import Indices, Kernel, Reference, add_into, assign_into, jump_channels, noise_rows
from ._problem import Accelerated, JumpChannels, Vector, VectorFunction

AUTO_MEMBERS = 1000


@dataclass
class Lowering:
    """Numeric functions of a form produced by a backend.

    Args:
        drift:
            The drift function, or `None`.

        diffusion:
            The function that returns every Wiener process amplitude, or `None`.

        noise_rows:
            The state index each Wiener process drives.

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
    propensities: VectorFunction | None = None
    jumps: tuple[JumpChannels, ...] = ()
    update: VectorFunction | None = None
    boundary: VectorFunction | None = None
    accelerated: Accelerated | None = None


class Backend(BaseEstimator):
    """Base class of backends, which turn kernels into the numeric functions of a form.

    Subclasses set the class variable `name` and implement `lower`.
    """

    name: ClassVar[str]

    @abstractmethod
    def lower(
        self,
        form: Form,
        kernels: Mapping[type, list[Kernel]],
        changes: Mapping[int, list[tuple[Reference, float]]],
        size: int,
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


class NumpyBackend(Backend):
    """Reference backend that evaluates every kernel with NumPy array operations.

    Examples:
        >>> from skcomplex.core.compile import NumpyBackend, compile_system
        >>> from skcomplex.core.model import System
        >>> from skcomplex.models.biology.ecology import Population
        >>> compile_system(System(elements=[Population('a')]), backend=NumpyBackend()).backend
        'numpy'
    """

    name: ClassVar[str] = 'numpy'

    def lower(
        self,
        form: Form,
        kernels: Mapping[type, list[Kernel]],
        changes: Mapping[int, list[tuple[Reference, float]]],
        size: int,
    ) -> Lowering:
        """Return the numeric functions of a form evaluated with NumPy.

        Args:
            form:
                The form of the problem.

            kernels:
                The kernels of every law kind, in evaluation order.

            changes:
                The changes of every jump output, keyed by the output's identity.

            size:
                The size of the state vector.

        Returns:
            lowering:
                The numeric functions.
        """
        lowering = Lowering()
        if kernels.get(Boundary):
            lowering.boundary = assigning_function(kernels[Boundary])
        if form in (Form.ODE, Form.SDE):
            lowering.drift = adding_function(kernels.get(Rate, []), size)
        if form is Form.SDE:
            lowering.diffusion = concatenating_function(kernels.get(Noise, []))
            lowering.noise_rows = noise_rows(kernels.get(Noise, []))
        if form is Form.JUMP:
            lowering.propensities = concatenating_function(kernels.get(Jump, []))
            lowering.jumps = jump_channels(kernels.get(Jump, []), changes)
        if form is Form.DISCRETE:
            lowering.update = assigning_function(kernels.get(Update, []))
        return lowering


BACKENDS = Registry[type[Backend]]('backend')


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
    name = ('numba' if members >= AUTO_MEMBERS else 'numpy') if backend == 'auto' else backend
    try:
        return BACKENDS.resolve(name)()
    except KeyError as error:
        names = ', '.join(['auto', *map(str, BACKENDS)])
        error_msg = f'No backend named {backend!r}. Available backends are: {names}.'
        raise BackendError(error_msg) from error


register_backend(NumpyBackend)
