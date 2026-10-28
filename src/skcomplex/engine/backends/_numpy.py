"""Run compiled problems in NumPy, the default backend, evaluating kernels as vectorised array operations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import ClassVar

from ...core.laws import (
    BoundaryLaw,
    CrossingLaw,
    DestructionLaw,
    DissipatorLaw,
    DriveLaw,
    EmissionLaw,
    EncounterLaw,
    EquationLaw,
    ExtremalLaw,
    FluxLaw,
    FractionalNoiseLaw,
    GeneratorLaw,
    GraphProductionLaw,
    JumpLaw,
    LocalRuleLaw,
    MatchLaw,
    NoiseLaw,
    ProductionLaw,
    ProliferationLaw,
    RateLaw,
    RelaxationLaw,
    RuleLaw,
    SampleLaw,
    SubstitutionLaw,
    TransferLaw,
    TransitionLaw,
    UpdateLaw,
)
from ...core.spaces import (
    CayleySpace,
    ContinuousSpace,
    ContinuousTime,
    DiscreteTime,
    LatticeSpace,
    MeshSpace,
    NetworkSpace,
    StaticTime,
)
from ..problems import (
    Draws,
    Form,
    Kernel,
    NoiseKind,
    OrderKind,
    ReductionSchedule,
    Reference,
    SupportSpec,
)
from ._base import BaseBackend
from ._lowering import _FUNCTION_KINDS, Lowering


class NumpyBackend(BaseBackend):
    """Reference backend that evaluates every kernel with NumPy array operations.

    Examples:
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.engine.backends import NumpyBackend
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.ecology import PopulationElement
        >>> compile_system(System(elements=[PopulationElement('a')]), backend=NumpyBackend()).backend
        'numpy'
    """

    supports: ClassVar[SupportSpec] = SupportSpec(
        time={ContinuousTime, DiscreteTime, StaticTime},
        space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
        laws={
            BoundaryLaw,
            CrossingLaw,
            DestructionLaw,
            DissipatorLaw,
            DriveLaw,
            EmissionLaw,
            EncounterLaw,
            EquationLaw,
            ExtremalLaw,
            FluxLaw,
            FractionalNoiseLaw,
            GeneratorLaw,
            GraphProductionLaw,
            JumpLaw,
            LocalRuleLaw,
            MatchLaw,
            NoiseLaw,
            ProductionLaw,
            ProliferationLaw,
            RateLaw,
            RelaxationLaw,
            RuleLaw,
            SampleLaw,
            SubstitutionLaw,
            TransferLaw,
            TransitionLaw,
            UpdateLaw,
        },
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
            lowering.boundary = BaseBackend._assigning_function(kernels[BoundaryLaw])
        if form in (Form.ODE, Form.FRACTIONAL, Form.SDE):
            lowering.drift = BaseBackend._adding_function(kernels.get(RateLaw, []), size)
        if form == Form.SDE:
            lowering.diffusion = BaseBackend._concatenating_function(kernels.get(NoiseLaw, []))
            noise = Kernel._lower_noise(kernels.get(NoiseLaw, []))
            lowering.noise_rows, lowering.noise_columns, lowering.n_processes, lowering.noise_blocks = noise
        if form == Form.JUMP:
            lowering.propensities = BaseBackend._concatenating_function(kernels.get(JumpLaw, []))
            lowering.jumps = Kernel._jump_channels(kernels.get(JumpLaw, []), changes)
        if form == Form.DISCRETE:
            lowering.update = BaseBackend._assigning_function(kernels.get(UpdateLaw, []))
        return self._with_reductions(lowering, reductions, kernels)

    @staticmethod
    def _with_reductions(
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
        for name, kind in _FUNCTION_KINDS.items():
            if kernels is None or BaseBackend._reads_reductions(kernels.get(kind, [])):
                setattr(lowering, name, reductions.wrap(getattr(lowering, name)))
        return lowering
