"""Realizations shared by the physics packages: the field's measures, and, by family, particles and quantum species."""

from ._avalanches import AvalancheAnalysis
from ._crackling_noise import CracklingNoiseAnalysis
from ._kuramoto_order import KuramotoOrderAnalysis
from ._milling_order import MillingOrderAnalysis
from ._nematic_order import NematicOrderAnalysis
from ._polar_order import PolarOrderAnalysis

__all__: list[str] = [
    'AvalancheAnalysis',
    'CracklingNoiseAnalysis',
    'KuramotoOrderAnalysis',
    'MillingOrderAnalysis',
    'NematicOrderAnalysis',
    'PolarOrderAnalysis',
]
