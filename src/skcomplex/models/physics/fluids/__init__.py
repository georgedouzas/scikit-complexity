"""Fluid dynamics: flows, convection cells, turbulence, shocks, shallow water, wave trains, and kinetic gases."""

from ._flows import (
    BoussinesqComponent,
    BurgersComponent,
    KuramotoSivashinskyComponent,
    ShallowWaterComponent,
    VorticityComponent,
    burgers,
    decaying_turbulence,
    incompressible_flow,
    kuramoto_sivashinsky,
    rayleigh_benard,
    shallow_water,
)
from ._kinetics import CarlemanGasComponent, WaveNumberComponent, discrete_velocity_boltzmann, wave_breaking
from ._lattice_gases import FHPSiteComponent, fhp_lattice_gas
from ._marangoni import MarangoniLayerComponent, benard_marangoni

__all__: list[str] = [
    'BoussinesqComponent',
    'BurgersComponent',
    'CarlemanGasComponent',
    'FHPSiteComponent',
    'KuramotoSivashinskyComponent',
    'MarangoniLayerComponent',
    'ShallowWaterComponent',
    'VorticityComponent',
    'WaveNumberComponent',
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
