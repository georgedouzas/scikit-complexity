"""Define elements, the parts of a system that hold its values.

An element holds the values of one thing or of a group of members, such as the height and mass of a ball in
`Element('ball', values={'x': 10.0, 'm': 1.0})`, and says how a visualisation observer draws it. Specs written in
`Annotated` mark a value as an angle, a conserved quantity, a matrix, or a link to a member of another group.
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
