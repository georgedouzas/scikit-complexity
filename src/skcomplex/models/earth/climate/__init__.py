"""Climate: energy balance, variability, glacial cycles, low-order atmospheres, net energy, and tipping."""

from ._energy_balance import (
    DaisyworldElement,
    IceAlbedoClimateElement,
    LatitudeBandElement,
    budyko_sellers,
    daisyworld,
    energy_balance,
)
from ._ice_ages import (
    CarbonCycleElement,
    SaltzmanMaaschElement,
    VanDerPolIceAgesElement,
    carbon_cycle,
    saltzman_glaciation,
    van_der_pol_ice_ages,
)
from ._low_order import GeodynamoElement, Lorenz84Element, Lorenz96Element, geodynamo, lorenz_84, lorenz_96
from ._net_energy import energy_return
from ._soil import SoilTemperatureElement, soil_temperature
from ._tipping import TippingElementElement, tipping_cascade_network
from ._tipping_types import (
    RampedFoldElement,
    StochasticEnergyBalanceElement,
    ramped_stochastic_fold,
    sutera_fraedrich_tipping,
)
from ._variability import ClimateAnomalyElement, DelayedOscillatorElement, enso, hasselmann

__all__: list[str] = [
    'CarbonCycleElement',
    'ClimateAnomalyElement',
    'DaisyworldElement',
    'DelayedOscillatorElement',
    'GeodynamoElement',
    'IceAlbedoClimateElement',
    'LatitudeBandElement',
    'Lorenz84Element',
    'Lorenz96Element',
    'RampedFoldElement',
    'SaltzmanMaaschElement',
    'SoilTemperatureElement',
    'StochasticEnergyBalanceElement',
    'TippingElementElement',
    'VanDerPolIceAgesElement',
    'budyko_sellers',
    'carbon_cycle',
    'daisyworld',
    'energy_balance',
    'energy_return',
    'enso',
    'geodynamo',
    'hasselmann',
    'lorenz_84',
    'lorenz_96',
    'ramped_stochastic_fold',
    'saltzman_glaciation',
    'soil_temperature',
    'sutera_fraedrich_tipping',
    'tipping_cascade_network',
    'van_der_pol_ice_ages',
]
