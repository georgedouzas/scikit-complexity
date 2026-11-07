"""Climate: energy balance, variability, glacial cycles, low-order atmospheres, net energy, forests, and tipping."""

from ._early_warning import EarlyWarningAnalysis
from ._energy_balance import (
    DaisyworldElement,
    IceAlbedoClimateElement,
    LatitudeBandElement,
    budyko_sellers,
    daisyworld,
    energy_balance,
)
from ._forests import ForestCoverElement, MitigatorsElement, forest_dieback, social_forest_dieback
from ._ice_ages import (
    CarbonCycleElement,
    SaltzmanMaaschElement,
    VanDerPolIceAgesElement,
    carbon_cycle,
    saltzman_glaciation,
    van_der_pol_ice_ages,
)
from ._low_order import GeodynamoElement, Lorenz84Element, Lorenz96Element, geodynamo, lorenz_84, lorenz_96
from ._net_energy import EnergySectorElement, energy_return
from ._pollution import PollutedRegionElement, population_pollution
from ._soil import SoilTemperatureElement, soil_temperature
from ._tipping import TippingElementElement, tipping_cascade_network
from ._tipping_types import (
    FoldFollowerElement,
    HopfFollowerElement,
    RampedFoldElement,
    StochasticEnergyBalanceElement,
    leading_following_cascade,
    ramped_stochastic_fold,
    sutera_fraedrich_tipping,
)
from ._variability import ClimateAnomalyElement, DelayedOscillatorElement, enso, hasselmann

__all__: list[str] = [
    'CarbonCycleElement',
    'ClimateAnomalyElement',
    'DaisyworldElement',
    'DelayedOscillatorElement',
    'EarlyWarningAnalysis',
    'EnergySectorElement',
    'FoldFollowerElement',
    'ForestCoverElement',
    'GeodynamoElement',
    'HopfFollowerElement',
    'IceAlbedoClimateElement',
    'LatitudeBandElement',
    'Lorenz84Element',
    'Lorenz96Element',
    'MitigatorsElement',
    'PollutedRegionElement',
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
    'forest_dieback',
    'geodynamo',
    'hasselmann',
    'leading_following_cascade',
    'lorenz_84',
    'lorenz_96',
    'population_pollution',
    'ramped_stochastic_fold',
    'saltzman_glaciation',
    'social_forest_dieback',
    'soil_temperature',
    'sutera_fraedrich_tipping',
    'tipping_cascade_network',
    'van_der_pol_ice_ages',
]
