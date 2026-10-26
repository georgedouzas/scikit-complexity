"""Run compiled problems in NumPy, the default backend, evaluating kernels as vectorised array operations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import ClassVar

from ...core.base import DomainKind, ElementKind, Form, LawKind, NoiseKind, OrderKind, SpaceKind, Supports, TimeKind
from ...core.laws import BoundaryLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ...core.problems import BaseBackend, Draws, Kernel, Lowering, ReductionSchedule, Reference, register_backend
from ...core.problems._backends import adding_function, assigning_function, concatenating_function, with_reductions
from ...core.problems._kernels import jump_channels, lower_noise


class NumpyBackend(BaseBackend):
    """Reference backend that evaluates every kernel with NumPy array operations.

    Examples:
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.engine.backends import NumpyBackend
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.ecology import PopulationElement
        >>> compile_system(System(components=[PopulationElement('a')]), backend=NumpyBackend()).backend
        'numpy'
    """

    supports: ClassVar[Supports] = Supports(
        time=set(TimeKind),
        space=set(SpaceKind.registered()),
        laws=set(LawKind.registered()),
        domains=set(DomainKind),
        components=set(ElementKind),
        noise=set(NoiseKind),
        orders=set(OrderKind),
    )

    name: ClassVar[str] = 'numpy'

    def _run(
        self,
        kernels: Mapping[type, list[Kernel]],
        changes: Mapping[int, list[tuple[Reference, float]]],
        *,
        reductions: ReductionSchedule | None = None,
        draws: Draws | None = None,
    ) -> Lowering:
        """Return the numeric functions of the prepared form evaluated with NumPy.

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
        """
        del draws
        form, size = self.form_, self.size_
        lowering = Lowering()
        if kernels.get(BoundaryLaw):
            lowering.boundary = assigning_function(kernels[BoundaryLaw])
        if form in (Form.ODE, Form.FRACTIONAL, Form.SDE):
            lowering.drift = adding_function(kernels.get(RateLaw, []), size)
        if form == Form.SDE:
            lowering.diffusion = concatenating_function(kernels.get(NoiseLaw, []))
            noise = lower_noise(kernels.get(NoiseLaw, []))
            lowering.noise_rows, lowering.noise_columns, lowering.n_processes, lowering.noise_blocks = noise
        if form == Form.JUMP:
            lowering.propensities = concatenating_function(kernels.get(JumpLaw, []))
            lowering.jumps = jump_channels(kernels.get(JumpLaw, []), changes)
        if form == Form.DISCRETE:
            lowering.update = assigning_function(kernels.get(UpdateLaw, []))
        return with_reductions(lowering, reductions, kernels)


register_backend(NumpyBackend)
