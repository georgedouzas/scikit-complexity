"""Assets shared by the economics packages.

They are assets with a price and the geometric Brownian motion of a price.
"""

from ._asset import AssetElement
from ._geometric_brownian_motion import GeometricBrownianMotionInteraction

__all__: list[str] = [
    'AssetElement',
    'GeometricBrownianMotionInteraction',
]
