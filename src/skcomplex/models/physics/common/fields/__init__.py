"""Field realizations shared by the physics packages: phase fields, free energies, and tensor transports."""

from ._landau_energy import LandauEnergyInteraction
from ._phase_field import PhaseFieldElement
from ._swift_hohenberg_energy import SwiftHohenbergEnergyInteraction
from ._transports import corotational, upper_convected

__all__: list[str] = [
    'LandauEnergyInteraction',
    'PhaseFieldElement',
    'SwiftHohenbergEnergyInteraction',
    'corotational',
    'upper_convected',
]
