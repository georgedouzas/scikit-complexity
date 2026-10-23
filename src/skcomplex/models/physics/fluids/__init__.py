"""Fluid dynamics: flows, convection cells, turbulence, shocks, shallow water, wave trains, and kinetic gases."""

from ._flows import (
    Boussinesq,
    Burgers,
    KuramotoSivashinsky,
    ShallowWater,
    Vorticity,
    burgers,
    decaying_turbulence,
    incompressible_flow,
    kuramoto_sivashinsky,
    rayleigh_benard,
    shallow_water,
)
from ._kinetics import CarlemanGas, WaveNumber, discrete_velocity_boltzmann, wave_breaking
from ._lattice_gases import FHPSite, fhp_lattice_gas
from ._marangoni import MarangoniLayer, benard_marangoni

__all__: list[str] = [
    'Boussinesq',
    'Burgers',
    'CarlemanGas',
    'FHPSite',
    'KuramotoSivashinsky',
    'MarangoniLayer',
    'ShallowWater',
    'Vorticity',
    'WaveNumber',
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
