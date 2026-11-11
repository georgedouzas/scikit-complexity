"""Define elements, the components of a system that hold its states.

An element holds the states of a system, such as an amount, a position, or a word. A group holds many elements of one
class as arrays, and a field holds one at every site of a space.
"""

from ._base import BaseElement
from ._declared import DeclaredElement
from ._element import Element
from ._field import FieldElement
from ._group import GroupElement
from ._patch import PatchElement
from ._word import WordElement

__all__: list[str] = [
    'BaseElement',
    'DeclaredElement',
    'Element',
    'FieldElement',
    'GroupElement',
    'PatchElement',
    'WordElement',
]
