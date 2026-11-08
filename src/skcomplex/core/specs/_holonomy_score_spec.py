"""Define the holonomy score spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec
from ._matrix_group_spec import MatrixGroupSpec


@dataclass(frozen=True)
class HolonomyScoreSpec(BaseSpec):
    """Spec of an objective of a matrix group state on the edges of a cell complex.

    The objective is a sum over the holonomies of faces. The objective is the sum over the faces of a class function of
    the holonomy of every face, the path-ordered product of the matrices of the edges around it. A class function, such
    as the real part of the trace, has the same value on every cyclic order of the product, so the local objective of an
    edge sums it over the holonomies of the faces around the edge that start at the edge.

    Args:
        function:
            The function `function(holonomies, **parameters)` of an array of holonomies, matrices on the last two axes,
            and the parameters of the system that it names, which returns one number per holonomy.

        group:
            The matrix group of the state.

        hits:
            The number of moves proposed per edge and sweep.

    Raises:
        ValidationError:
            The function is not callable, the group is invalid, or the number of hits is not positive.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.specs import HolonomyScoreSpec, MatrixGroupSpec
        >>> trace = lambda holonomies: -np.trace(holonomies, axis1=-2, axis2=-1).real
        >>> HolonomyScoreSpec(trace, MatrixGroupSpec('SU', 2), hits=4).hits
        4
    """

    function: Callable[..., object]
    group: MatrixGroupSpec
    hits: int = 1

    def _check(self) -> None:
        """Raise an error for a function not callable, a bad group, or a bad hit count.

        The number of hits must be at least 1.

        Raises:
            ValidationError:
                The function is not callable, the group is invalid, or the number of hits is not positive.
        """
        if not callable(self.function):
            error_msg = f'The function of a holonomy score must be callable, but it is {self.function!r}.'
            raise ValidationError(error_msg)
        if not isinstance(self.group, MatrixGroupSpec):
            error_msg = f'The group of a holonomy score is a MatrixGroupSpec, but it is {self.group!r}.'
            raise ValidationError(error_msg)
        self.group._validated()
        if not isinstance(self.hits, int) or self.hits < 1:
            error_msg = f'A holonomy score proposes at least one move per edge, but it has hits={self.hits!r}.'
            raise ValidationError(error_msg)
