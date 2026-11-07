"""Climate: energy balance, variability, glacial cycles, low-order atmospheres, net energy, forests, and tipping."""

from ._early_warning import EarlyWarningResult, early_warning
from ._energy_balance import (
    BudykoSellersSystem,
    DaisyworldElement,
    DaisyworldSystem,
    EnergyBalanceSystem,
    IceAlbedoClimateElement,
    LatitudeBandElement,
)
from ._forests import ForestCoverElement, ForestDiebackSystem, MitigatorsElement, SocialForestDiebackSystem
from ._ice_ages import (
    CarbonCycleElement,
    CarbonCycleSystem,
    SaltzmanGlaciationSystem,
    SaltzmanMaaschElement,
    VanDerPolIceAgesElement,
    VanDerPolIceAgesSystem,
)
from ._low_order import (
    GeodynamoElement,
    GeodynamoSystem,
    Lorenz84Element,
    Lorenz84System,
    Lorenz96Element,
    Lorenz96System,
)
from ._net_energy import EnergyReturnSystem, EnergySectorElement
from ._pollution import PollutedRegionElement, PopulationPollutionSystem
from ._soil import SoilTemperatureElement, SoilTemperatureSystem
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
from ._variability import ClimateAnomalyElement, DelayedOscillatorElement, ENSOSystem, HasselmannSystem

__all__: list[str] = [
    'BudykoSellersSystem',
    'CarbonCycleElement',
    'CarbonCycleSystem',
    'ClimateAnomalyElement',
    'DaisyworldElement',
    'DaisyworldSystem',
    'DelayedOscillatorElement',
    'ENSOSystem',
    'EarlyWarningResult',
    'EnergyBalanceSystem',
    'EnergyReturnSystem',
    'EnergySectorElement',
    'FoldFollowerElement',
    'ForestCoverElement',
    'ForestDiebackSystem',
    'GeodynamoElement',
    'GeodynamoSystem',
    'HasselmannSystem',
    'HopfFollowerElement',
    'IceAlbedoClimateElement',
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
    'SoilTemperatureElement',
    'SoilTemperatureSystem',
    'StochasticEnergyBalanceElement',
    'SuteraFraedrichTippingSystem',
    'TippingCascadeNetworkSystem',
    'TippingElementElement',
    'VanDerPolIceAgesElement',
    'VanDerPolIceAgesSystem',
    'early_warning',
]
