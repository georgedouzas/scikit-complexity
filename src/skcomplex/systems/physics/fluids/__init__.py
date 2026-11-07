"""Fluid dynamics: flows, convection cells, turbulence, shocks, shallow water, wave trains, and kinetic gases."""

from ._flows import (
    BoussinesqElement,
    BurgersElement,
    BurgersSystem,
    DecayingTurbulenceSystem,
    KuramotoSivashinskyElement,
    KuramotoSivashinskySystem,
    RayleighBenardSystem,
    ShallowWaterElement,
    ShallowWaterSystem,
    VorticityElement,
    incompressible_flow,
)
from ._kinetic_equations import (
    BGKEquationSystem,
    BhatnagarGrossKrookInteraction,
    BoltzmannEquationSystem,
    EnskogSystem,
    VlasovEquationSystem,
)
from ._kinetics import CarlemanGasElement, DiscreteVelocityBoltzmannSystem, WaveBreakingSystem, WaveNumberElement
from ._lattice_gases import FHPLatticeGasSystem, FHPSiteElement, HPPLatticeGasSystem, HPPSiteElement
from ._marangoni import BenardMarangoniSystem, MarangoniLayerElement

__all__: list[str] = [
    'BGKEquationSystem',
    'BenardMarangoniSystem',
    'BhatnagarGrossKrookInteraction',
    'BoltzmannEquationSystem',
    'BoussinesqElement',
    'BurgersElement',
    'BurgersSystem',
    'CarlemanGasElement',
    'DecayingTurbulenceSystem',
    'DiscreteVelocityBoltzmannSystem',
    'EnskogSystem',
    'FHPLatticeGasSystem',
    'FHPSiteElement',
    'HPPLatticeGasSystem',
    'HPPSiteElement',
    'KuramotoSivashinskyElement',
    'KuramotoSivashinskySystem',
    'MarangoniLayerElement',
    'RayleighBenardSystem',
    'ShallowWaterElement',
    'ShallowWaterSystem',
    'VlasovEquationSystem',
    'VorticityElement',
    'WaveBreakingSystem',
    'WaveNumberElement',
    'incompressible_flow',
]
