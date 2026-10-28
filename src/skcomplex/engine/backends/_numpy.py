"""Run compiled problems in NumPy, the default backend, evaluating kernels as vectorised array operations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import replace

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
from ..problems import Form, NoiseKind, OrderKind, SupportSpec
from ..problems._law_kernel import LawKernel
from ..problems._lowered_kernels import LoweredKernels
from ..problems._reduction_schedule import ReductionSchedule
from ._base import BaseBackend
from ._lowered_functions import _FUNCTION_KINDS, LoweredFunctions


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

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the backend writes functions of."""
        return SupportSpec(
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

    @classmethod
    def _get_name(cls) -> str:
        """Return the name of the backend that problems record."""
        return 'numpy'

    def _run(self, inputs: LoweredKernels) -> LoweredFunctions:
        """Return the numeric functions of the form of the kernels, evaluated with NumPy."""
        form, size, kernels, changes = inputs.form, inputs.size, inputs.kernels, inputs.jump_changes
        functions: dict[str, object] = {}
        if kernels.get(BoundaryLaw):
            functions['boundary'] = BaseBackend._assigning_function(kernels[BoundaryLaw])
        if form in (Form.ODE, Form.FRACTIONAL, Form.SDE):
            functions['drift'] = BaseBackend._adding_function(kernels.get(RateLaw, []), size)
        if form == Form.SDE:
            functions['diffusion'] = BaseBackend._concatenating_function(kernels.get(NoiseLaw, []))
            noise = LawKernel._lower_noise(kernels.get(NoiseLaw, []))
            names = ('noise_rows', 'noise_columns', 'n_processes', 'noise_blocks')
            functions.update(zip(names, noise, strict=True))
        if form == Form.JUMP:
            functions['propensities'] = BaseBackend._concatenating_function(kernels.get(JumpLaw, []))
            functions['jumps'] = LawKernel._jump_channels(kernels.get(JumpLaw, []), changes)
        if form == Form.DISCRETE:
            functions['update'] = BaseBackend._assigning_function(kernels.get(UpdateLaw, []))
        return self._with_reductions(LoweredFunctions(**functions), inputs.reductions, kernels)

    @staticmethod
    def _with_reductions(
        lowering: LoweredFunctions,
        reductions: ReductionSchedule | None,
        kernels: Mapping[type, list[LawKernel]] | None = None,
    ) -> LoweredFunctions:
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
        wrapped = {
            name: reductions.wrap(getattr(lowering, name))
            for name, kind in _FUNCTION_KINDS.items()
            if kernels is None or any(kernel.reads_reductions for kernel in kernels.get(kind, []))
        }
        return replace(lowering, **wrapped)
