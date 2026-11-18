"""Define the symmetry of a tensor value of an element."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...enums import SymmetryKind
from ...errors import ValidationError
from ...specs._base_spec import BaseSpec


@dataclass(frozen=True)
class Tensor(BaseSpec):
    """A symmetry of a value that holds a square matrix, such as an order tensor.

    It is written in `Annotated` beside the allowed values, such as `Q: Annotated[object, Tensor('traceless')] = 0.0`.
    The value holds the independent components of the matrix, row after row, as `Q_xx` and `Q_xy` of a traceless
    tensor in two dimensions, and the matrix has one row per dimension of the space of the element. Laws read the value
    as a Sage matrix, so `s.target.Q.trace()` and `s.target.Q * s.target.Q` work.

    Unlike a list of numbers, which is a vector, it says the numbers are the entries of a matrix of a symmetry, so a
    traceless matrix stays traceless. Choose it when a value is an order tensor, a stress, or a strain.

    It appears in many fields. The alignment of a liquid crystal, the stress of a gel, and the strain of a rock are
    each a tensor.

    Args:
        symmetry:
            `'full'`, `'symmetric'`, `'antisymmetric'`, or `'traceless'`, which means symmetric and traceless.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element, Tensor
        >>> class NematicElement(Element):
        ...     def __init__(self, label, Q: Annotated[object, Tensor('traceless')] = 0.0, *, space=None):
        ...         super().__init__(label, space=space)
        ...         self.Q = Q
        >>> Tensor('traceless').symmetry
        'traceless'
    """

    symmetry: str

    def _check(self) -> None:
        """Raise an error for an unknown symmetry.

        Raises:
            ValidationError:
                The symmetry is not one of the kinds of tensor.
        """
        if self.symmetry not in SymmetryKind:
            error_msg = (
                f"A tensor has one of the symmetries {', '.join(SymmetryKind)}, such as Tensor('traceless'), but it is "
                f'{self.symmetry!r}.'
            )
            raise ValidationError(error_msg)
