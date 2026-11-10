"""Define the elements that the systems of several fields share.

They hold one nonnegative amount, or the position of a point in a continuous space.
"""

from ._nonnegative import NonnegativeElement
from ._point import PointElement

__all__: list[str] = [
    'NonnegativeElement',
    'PointElement',
]
