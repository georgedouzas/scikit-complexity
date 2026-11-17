"""Climate systems.

The package covers energy balance, variability, glacial cycles, low-order atmospheres, net energy, forests, and tipping.
"""

from ._early_warning import EarlyWarningResult, early_warning
from ._energy_balance import (
    BudykoSellersSystem,
    DaisyworldSystem,
    EnergyBalanceSystem,
    IceAlbedoBalanceInteraction,
    LatitudeBandElement,
    LatitudeEnergyBalanceInteraction,
)
from ._forests import (
    ForestCoverElement,
    ForestDiebackSystem,
    ForestGrowthInteraction,
    MitigatedForestGrowthInteraction,
    MitigationInteraction,
    MitigatorsElement,
    SocialForestDiebackSystem,
)
from ._ice_ages import (
    CarbonCycleInteraction,
    CarbonCycleSystem,
    SaltzmanGlaciationSystem,
    SaltzmanMaaschElement,
    SaltzmanMaaschInteraction,
    VanDerPolIceAgesInteraction,
    VanDerPolIceAgesSystem,
)
from ._low_order import (
    GeodynamoInteraction,
    GeodynamoSystem,
    Lorenz84Element,
    Lorenz84Interaction,
    Lorenz84System,
    Lorenz96Element,
    Lorenz96System,
)
from ._net_energy import EnergyReturnSystem, EnergySectorElement
from ._pollution import PopulationPollutionSystem
from ._soil import SoilTemperatureSystem
from ._tipping import TippingCascadeInteraction, TippingCascadeNetworkSystem, TippingElementElement
from ._tipping_types import (
    FoldFollowerElement,
    FoldFollowerInteraction,
    HopfFollowerElement,
    HopfFollowerInteraction,
    LeadingFollowingCascadeSystem,
    RampedFoldElement,
    RampedFoldInteraction,
    RampedStochasticFoldSystem,
    StochasticEnergyBalanceInteraction,
    SuteraFraedrichTippingSystem,
)
from ._variability import (
    ClimateAnomalyElement,
    ClimateFeedbackInteraction,
    DelayedOscillatorInteraction,
    ENSOSystem,
    HasselmannSystem,
)

__all__: list[str] = [
    'BudykoSellersSystem',
    'CarbonCycleInteraction',
    'CarbonCycleSystem',
    'ClimateAnomalyElement',
    'ClimateFeedbackInteraction',
    'DaisyworldSystem',
    'DelayedOscillatorInteraction',
    'ENSOSystem',
    'EarlyWarningResult',
    'EnergyBalanceSystem',
    'EnergyReturnSystem',
    'EnergySectorElement',
    'FoldFollowerElement',
    'FoldFollowerInteraction',
    'ForestCoverElement',
    'ForestDiebackSystem',
    'ForestGrowthInteraction',
    'GeodynamoInteraction',
    'GeodynamoSystem',
    'HasselmannSystem',
    'HopfFollowerElement',
    'HopfFollowerInteraction',
    'IceAlbedoBalanceInteraction',
    'LatitudeBandElement',
    'LatitudeEnergyBalanceInteraction',
    'LeadingFollowingCascadeSystem',
    'Lorenz84Element',
    'Lorenz84Interaction',
    'Lorenz84System',
    'Lorenz96Element',
    'Lorenz96System',
    'MitigatedForestGrowthInteraction',
    'MitigationInteraction',
    'MitigatorsElement',
    'PopulationPollutionSystem',
    'RampedFoldElement',
    'RampedFoldInteraction',
    'RampedStochasticFoldSystem',
    'SaltzmanGlaciationSystem',
    'SaltzmanMaaschElement',
    'SaltzmanMaaschInteraction',
    'SocialForestDiebackSystem',
    'SoilTemperatureSystem',
    'StochasticEnergyBalanceInteraction',
    'SuteraFraedrichTippingSystem',
    'TippingCascadeInteraction',
    'TippingCascadeNetworkSystem',
    'TippingElementElement',
    'VanDerPolIceAgesInteraction',
    'VanDerPolIceAgesSystem',
    'early_warning',
]
