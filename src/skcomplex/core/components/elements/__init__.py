"""Define elements, the components of a system that hold its states.

An element holds the states of a system, such as an amount, a position, or a word, and says how a visualisation
observer draws it.
"""

from ._base import BaseElement
from ._element import Element
from ._member import Member
from ._period import Period
from ._quantity import Quantity
from ._tensor import Tensor
from ._visual_spec import VisualSpec

__all__: list[str] = [
    'BaseElement',
    'Element',
    'Member',
    'Period',
    'Quantity',
    'Tensor',
    'VisualSpec',
]
