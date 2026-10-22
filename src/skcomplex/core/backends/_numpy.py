"""Run compiled problems in NumPy, the default backend, evaluating kernels as vectorised array operations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import ClassVar

from ..base import ComponentKind, DomainKind, Form, LawKind, OrderKind, SpaceKind, Supports, TimeKind
from ..compile._backends import (
    Backend,
    Lowering,
    adding_function,
    assigning_function,
    concatenating_function,
    register_backend,
    with_reductions,
)
from ..compile._kernels import Kernel, Reference, jump_channels, noise_rows
from ..compile._problem import Draws
from ..compile._reductions import ReductionPass
from ..laws import Boundary, Jump, Noise, Rate, Update


class NumpyBackend(Backend):
    """Reference backend that evaluates every kernel with NumPy array operations.

    Examples:
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.core.backends import NumpyBackend
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.ecology import Population
        >>> compile_system(System(components=[Population('a')]), backend=NumpyBackend()).backend
        'numpy'
    """

    supports: ClassVar[Supports] = Supports(
        time=set(TimeKind),
        space=set(SpaceKind),
        laws=set(LawKind),
        domains=set(DomainKind),
        components=set(ComponentKind),
        orders=set(OrderKind),
    )

    name: ClassVar[str] = 'numpy'

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

            reductions:
                The reductions the kernels read, evaluated before them at every evaluation, or `None`.

            draws:
                The random values a stochastic map draws at every step into the parameter vector extended after its
                parameters, or `None` for a deterministic problem.

        Returns:
            lowering:
                The numeric functions.
        """
        del draws
        lowering = Lowering()
        if kernels.get(Boundary):
            lowering.boundary = assigning_function(kernels[Boundary])
        if form in (Form.ODE, Form.FRACTIONAL, Form.SDE):
            lowering.drift = adding_function(kernels.get(Rate, []), size)
        if form is Form.SDE:
            lowering.diffusion = concatenating_function(kernels.get(Noise, []))
            lowering.noise_rows = noise_rows(kernels.get(Noise, []))
        if form is Form.JUMP:
            lowering.propensities = concatenating_function(kernels.get(Jump, []))
            lowering.jumps = jump_channels(kernels.get(Jump, []), changes)
        if form is Form.DISCRETE:
            lowering.update = assigning_function(kernels.get(Update, []))
        return with_reductions(lowering, reductions)


register_backend(NumpyBackend)
