"""Fluid dynamics: flows, convection cells, turbulence, shocks, shallow water, wave trains, and kinetic gases."""

from ._bhatnagar_gross_krook import BhatnagarGrossKrookInteraction
from ._boussinesq import BoussinesqElement
from ._burgers import BurgersElement
from ._carleman_gas import CarlemanGasElement
from ._flows import (
    burgers,
    decaying_turbulence,
    incompressible_flow,
    kuramoto_sivashinsky,
    rayleigh_benard,
    shallow_water,
)
from ._hpp_site import HPPSiteElement
from ._kinetic_equations import bgk_equation, boltzmann_equation, enskog, vlasov_equation
from ._kinetics import discrete_velocity_boltzmann, wave_breaking
from ._kuramoto_sivashinsky import KuramotoSivashinskyElement
from ._lattice_gases import FHPSiteElement, fhp_lattice_gas, hpp_lattice_gas
from ._marangoni import MarangoniLayerElement, benard_marangoni
from ._shallow_water import ShallowWaterElement
from ._vorticity import VorticityElement
from ._wave_number import WaveNumberElement

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
