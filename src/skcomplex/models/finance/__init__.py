"""Finance: asset prices driven by geometric Brownian motion."""

from ._assets import Asset, GeometricBrownianMotion

__all__: list[str] = [
    'Asset',
    'GeometricBrownianMotion',
]
