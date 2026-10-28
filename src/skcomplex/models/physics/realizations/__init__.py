"""Realizations shared by the physics packages: the field's measures, and, by family, particles and quantum species."""

from ._avalanches import AvalancheAnalysis
from ._binder_cumulant import BinderCumulantAnalysis
from ._crackling_noise import CracklingNoiseAnalysis
from ._critical_exponent import CriticalExponentAnalysis
from ._giant_number_fluctuation import GiantNumberFluctuationAnalysis
from ._kinetic_temperature import KineticTemperatureAnalysis
from ._kuramoto_order import KuramotoOrderAnalysis
from ._maxwell_boltzmann import MaxwellBoltzmannAnalysis
from ._milling_order import MillingOrderAnalysis
from ._nematic_order import NematicOrderAnalysis
from ._polar_order import PolarOrderAnalysis
from ._replica_overlap import ReplicaOverlapAnalysis
from ._susceptibility import SusceptibilityAnalysis

__all__: list[str] = [
    'AvalancheAnalysis',
    'BinderCumulantAnalysis',
    'CracklingNoiseAnalysis',
    'CriticalExponentAnalysis',
    'GiantNumberFluctuationAnalysis',
    'KineticTemperatureAnalysis',
    'KuramotoOrderAnalysis',
    'MaxwellBoltzmannAnalysis',
    'MillingOrderAnalysis',
    'NematicOrderAnalysis',
    'PolarOrderAnalysis',
    'ReplicaOverlapAnalysis',
    'SusceptibilityAnalysis',
]
