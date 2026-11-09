"""Climate systems.

The package covers energy balance, variability, glacial cycles, low-order atmospheres, net energy, forests, and tipping.
"""

from ._early_warning import EarlyWarningResult, early_warning
from ._energy_balance import (
    BudykoSellersSystem,
    DaisyworldElement,
    DaisyworldSystem,
    EnergyBalanceSystem,
    LatitudeBandElement,
)
from ._forests import ForestCoverElement, ForestDiebackSystem, MitigatorsElement, SocialForestDiebackSystem
from ._ice_ages import (
    CarbonCycleSystem,
    SaltzmanGlaciationSystem,
    SaltzmanMaaschElement,
    VanDerPolIceAgesSystem,
)
from ._low_order import (
    GeodynamoSystem,
    Lorenz84Element,
    Lorenz84System,
    Lorenz96Element,
    Lorenz96System,
)
from ._net_energy import EnergyReturnSystem, EnergySectorElement
from ._pollution import PollutedRegionElement, PopulationPollutionSystem
from ._soil import SoilTemperatureSystem
from ._tipping import TippingCascadeNetworkSystem, TippingElementElement
from ._tipping_types import (
    FoldFollowerElement,
    HopfFollowerElement,
    LeadingFollowingCascadeSystem,
    RampedFoldElement,
    RampedStochasticFoldSystem,
    StochasticEnergyBalanceElement,
    SuteraFraedrichTippingSystem,
)
from ._variability import ClimateAnomalyElement, ENSOSystem, HasselmannSystem

__all__: list[str] = [
    'BudykoSellersSystem',
    'CarbonCycleSystem',
    'ClimateAnomalyElement',
    'DaisyworldElement',
    'DaisyworldSystem',
    'ENSOSystem',
    'EarlyWarningResult',
    'EnergyBalanceSystem',
    'EnergyReturnSystem',
    'EnergySectorElement',
    'FoldFollowerElement',
    'ForestCoverElement',
    'ForestDiebackSystem',
    'GeodynamoSystem',
    'HasselmannSystem',
    'HopfFollowerElement',
    'LatitudeBandElement',
    'LeadingFollowingCascadeSystem',
    'Lorenz84Element',
    'Lorenz84System',
    'Lorenz96Element',
    'Lorenz96System',
    'MitigatorsElement',
    'PollutedRegionElement',
    'PopulationPollutionSystem',
    'RampedFoldElement',
    'RampedStochasticFoldSystem',
    'SaltzmanGlaciationSystem',
    'SaltzmanMaaschElement',
    'SocialForestDiebackSystem',
    'SoilTemperatureSystem',
    'StochasticEnergyBalanceElement',
    'SuteraFraedrichTippingSystem',
    'TippingCascadeNetworkSystem',
    'TippingElementElement',
    'VanDerPolIceAgesSystem',
    'early_warning',
]
