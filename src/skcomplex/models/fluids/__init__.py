"""Fluid dynamics: incompressible flow, convection, turbulence, shocks, and shallow water."""

from ._flows import (
    Boussinesq,
    Burgers,
    ShallowWater,
    Vorticity,
    burgers,
    decaying_turbulence,
    incompressible_flow,
    rayleigh_benard,
    shallow_water,
)

__all__: list[str] = [
    'Boussinesq',
    'Burgers',
    'ShallowWater',
    'Vorticity',
    'burgers',
    'decaying_turbulence',
    'incompressible_flow',
    'rayleigh_benard',
    'shallow_water',
]
