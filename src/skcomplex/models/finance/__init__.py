"""Finance: asset prices driven by geometric Brownian motion."""

from ._assets import Asset, GeometricBrownianMotion
from ._factories import geometric_brownian_motion

__all__: list[str] = [
    'Asset',
    'GeometricBrownianMotion',
    'geometric_brownian_motion',
]
