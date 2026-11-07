"""Fluid dynamics: flows, convection cells, turbulence, shocks, shallow water, wave trains, and kinetic gases."""

from ._flows import (
    BoussinesqElement,
    BurgersElement,
    KuramotoSivashinskyElement,
    ShallowWaterElement,
    VorticityElement,
    burgers,
    decaying_turbulence,
    incompressible_flow,
    kuramoto_sivashinsky,
    rayleigh_benard,
    shallow_water,
)
from ._kinetic_equations import (
    BhatnagarGrossKrookInteraction,
    bgk_equation,
    boltzmann_equation,
    enskog,
    vlasov_equation,
)
from ._kinetics import CarlemanGasElement, WaveNumberElement, discrete_velocity_boltzmann, wave_breaking
from ._lattice_gases import FHPSiteElement, HPPSiteElement, fhp_lattice_gas, hpp_lattice_gas
from ._marangoni import MarangoniLayerElement, benard_marangoni

__all__: list[str] = [
    'BhatnagarGrossKrookInteraction',
    'BoussinesqElement',
    'BurgersElement',
    'CarlemanGasElement',
    'FHPSiteElement',
    'HPPSiteElement',
    'KuramotoSivashinskyElement',
    'MarangoniLayerElement',
    'ShallowWaterElement',
    'VorticityElement',
    'WaveNumberElement',
    'benard_marangoni',
    'bgk_equation',
    'boltzmann_equation',
    'burgers',
    'decaying_turbulence',
    'discrete_velocity_boltzmann',
    'enskog',
    'fhp_lattice_gas',
    'hpp_lattice_gas',
    'incompressible_flow',
    'kuramoto_sivashinsky',
    'rayleigh_benard',
    'shallow_water',
    'vlasov_equation',
    'wave_breaking',
]
