"""Define the symmetry of a tensor value of an element."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...base import BaseSpec
from ...enums import SymmetryKind
from ...errors import ValidationError


@dataclass(frozen=True)
class Tensor(BaseSpec):
    """A spec of a value that holds a square matrix of a symmetry, such as a stress.

    It is written in `Annotated` beside the allowed values, such as `Q: Annotated[object, Tensor('traceless')] = 0.0`. The
    value holds the independent entries of the matrix, row after row, such as `Q_xx` and `Q_xy` of a traceless matrix in two
    dimensions, and the matrix has one row per dimension of the space of the element. Laws read the value as a Sage matrix,
    so `s.target.Q.trace()` works, and a traceless matrix stays traceless. It holds the alignment of a liquid crystal, the
    stress of a gel, and the strain of a rock.

    Read more in the user guide, under Elements.

    Args:
        symmetry:
            `'full'`, `'symmetric'`, `'antisymmetric'`, or `'traceless'`, which means symmetric and traceless.

    See Also:
        Period: Marks a value that wraps around, such as an angle.

    Examples:
        >>> from skcomplex.core.components.elements import Tensor
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
