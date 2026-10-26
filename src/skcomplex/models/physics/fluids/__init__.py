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
from ._kinetics import CarlemanGasElement, WaveNumberElement, discrete_velocity_boltzmann, wave_breaking
from ._lattice_gases import FHPSiteElement, fhp_lattice_gas
from ._marangoni import MarangoniLayerElement, benard_marangoni

__all__: list[str] = [
    'BoussinesqElement',
    'BurgersElement',
    'CarlemanGasElement',
    'FHPSiteElement',
    'KuramotoSivashinskyElement',
    'MarangoniLayerElement',
    'ShallowWaterElement',
    'VorticityElement',
    'WaveNumberElement',
    'benard_marangoni',
    'burgers',
    'decaying_turbulence',
    'discrete_velocity_boltzmann',
    'fhp_lattice_gas',
    'incompressible_flow',
    'kuramoto_sivashinsky',
    'rayleigh_benard',
    'shallow_water',
    'wave_breaking',
]
