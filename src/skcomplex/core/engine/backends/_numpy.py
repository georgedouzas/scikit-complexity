"""Run compiled problems in NumPy, the default backend."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import replace

from ...components.laws import (
    ArrayUpdateLaw,
    BilinearLaw,
    BoundaryLaw,
    CascadeLaw,
    CrossingLaw,
    DissipatorLaw,
    EquationLaw,
    ExtremalLaw,
    FluxLaw,
    GeneratorLaw,
    JumpLaw,
    LocalRuleLaw,
    MatchLaw,
    NoiseLaw,
    RateLaw,
    RuleLaw,
    SampleLaw,
    TransferLaw,
    TransitionLaw,
    UpdateLaw,
)
from ...components.spaces import (
    CayleySpace,
    ContinuousSpace,
    ContinuousTime,
    DiscreteTime,
    LatticeSpace,
    MeshSpace,
    NetworkSpace,
    StaticTime,
)
from ...enums import FormKind, NoiseKind, OrderKind
from ...errors import ValidationError
from ...specs import LoweredFunctionsSpec, SupportSpec
from ...types import VectorFunction
from ..problems._law_kernel import LawKernel
from ..problems._lowered_kernels_spec import LoweredKernelsSpec
from ..problems._reduction_schedule_spec import ReductionScheduleSpec
from ._base import BaseBackend


class NumpyBackend(BaseBackend):
    """A backend that evaluates every kernel with NumPy array operations, the default.

    Choose it by default, for small systems, and to check the other backends. It appears in every field, as the
    reference that runs any law the compiler can lower.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.engine.backends import NumpyBackend
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.systems.biology.common.populations import PopulationElement
        >>> system = CompositeSystem(elements=[PopulationElement('a')])
        >>> simulate(system, until=0, backend=NumpyBackend()).problem.backend
        'numpy'
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return what the backend writes functions of.

        It is the times, spaces, laws, noise, and orders.
        """
        return SupportSpec(
            time={ContinuousTime, DiscreteTime, StaticTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={
                BoundaryLaw,
                CrossingLaw,
                DissipatorLaw,
                ArrayUpdateLaw,
                EquationLaw,
                ExtremalLaw,
                FluxLaw,
                GeneratorLaw,
                JumpLaw,
                LocalRuleLaw,
                MatchLaw,
                NoiseLaw,
                BilinearLaw,
                RateLaw,
                CascadeLaw,
                RuleLaw,
                SampleLaw,
                TransferLaw,
                TransitionLaw,
                UpdateLaw,
            },
            noise=frozenset(NoiseKind),
            orders=frozenset(OrderKind),
        )

    @classmethod
    def _get_name(cls) -> str:
        """Return the name of the backend that problems record."""
        return 'numpy'

    def _run(self, inputs: LoweredKernelsSpec | None) -> None:
        """Set the numeric functions of the form of the kernels, evaluated with NumPy.

        Args:
            inputs:
                The kernels of a problem.

        Raises:
            ValidationError:
                No kernels are passed.
        """
        if inputs is None:
            error_msg = 'A backend runs on the kernels of a problem. Pass a `LoweredKernelsSpec` to `run`.'
            raise ValidationError(error_msg)
        form, size, kernels = inputs.form, inputs.size, inputs.kernels
        boundary = kernels.get(BoundaryLaw)
        lowering = LoweredFunctionsSpec(
            drift=(
                BaseBackend._get_adding_function(kernels.get(RateLaw, []), size)
                if form in (FormKind.ODE, FormKind.FRACTIONAL, FormKind.SDE, FormKind.STATIC)
                else None
            ),
            diffusion=(
                BaseBackend._get_concatenating_function(kernels.get(NoiseLaw, [])) if form == FormKind.SDE else None
            ),
            propensities=(
                BaseBackend._get_concatenating_function(kernels.get(JumpLaw, [])) if form == FormKind.JUMP else None
            ),
            jumps=(
                LawKernel._jump_channels(kernels.get(JumpLaw, []), inputs.jump_changes) if form == FormKind.JUMP else ()
            ),
            update=(
                BaseBackend._get_assigning_function(kernels.get(UpdateLaw, []))
                if form in (FormKind.DISCRETE, FormKind.STATIC)
                else None
            ),
            boundary=BaseBackend._get_assigning_function(boundary) if boundary else None,
        )
        if form == FormKind.SDE:
            rows, columns, processes, blocks = LawKernel._lower_noise(kernels.get(NoiseLaw, []))
            lowering = replace(
                lowering,
                noise_rows=rows,
                noise_columns=columns,
                n_processes=processes,
                noise_blocks=blocks,
            )
        self.functions_ = self._copy_with_reductions(lowering, inputs.reductions, kernels)

    @staticmethod
    def _copy_with_reductions(
        lowering: LoweredFunctionsSpec,
        reductions: ReductionScheduleSpec | None,
        kernels: Mapping[type, list[LawKernel]] | None = None,
    ) -> LoweredFunctionsSpec:
        """Return a lowering whose functions evaluate the reductions before their kernels.

        The reductions are evaluated at the state they receive.

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
        return replace(
            lowering,
            drift=NumpyBackend._get_wrapped(lowering.drift, RateLaw, reductions, kernels),
            diffusion=NumpyBackend._get_wrapped(lowering.diffusion, NoiseLaw, reductions, kernels),
            propensities=NumpyBackend._get_wrapped(lowering.propensities, JumpLaw, reductions, kernels),
            update=NumpyBackend._get_wrapped(lowering.update, UpdateLaw, reductions, kernels),
            boundary=NumpyBackend._get_wrapped(lowering.boundary, BoundaryLaw, reductions, kernels),
        )

    @staticmethod
    def _get_wrapped(
        function: VectorFunction | None,
        kind: type,
        reductions: ReductionScheduleSpec,
        kernels: Mapping[type, list[LawKernel]] | None,
    ) -> VectorFunction | None:
        """Return a function that evaluates the reductions before it.

        It returns the function itself when its kernels read none.

        Args:
            function:
                The function of the extended parameter vector, or `None`.

            kind:
                The law kind of the kernels of the function.

            reductions:
                The reductions.

            kernels:
                The kernels of every law kind, or `None` to wrap every function.

        Returns:
            function:
                The function of the parameter vector, or `None`.
        """
        if kernels is None or any(kernel.reads_reductions for kernel in kernels.get(kind, [])):
            return reductions.wrap(function)
        return function
