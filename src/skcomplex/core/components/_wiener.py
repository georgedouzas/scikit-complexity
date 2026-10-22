"""Define Wiener elements, sets of Wiener processes with a constant covariance that noise laws share."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from typing import Any, ClassVar

from ..base import Expression, Requires, SymbolicVector, TimeKind, symbol, vector
from ._components import Element


class Wiener(Element):
    """Set of Wiener processes whose increments over a step `h` have the covariance `h C`.

    The element has no states. An interaction binds it to a role, and its noise laws name its processes through the
    handle `s.<role>.dW`, a symbol for one process and a vector of symbols for several, so `s.<role>.dW[k]` names
    process `k`. Laws of any component that name one process share its increment, and group laws that name it give all
    members one common process. The covariance is fixed when the system is compiled.

    Args:
        label:
            The label of the element.

        dimension:
            The number of processes.

        covariance:
            The constant covariance matrix `C` of the processes, symmetric and positive semidefinite of size
            `dimension`, or `None` for the identity.

    Examples:
        >>> from skcomplex.core.components import Wiener
        >>> Wiener('market', dimension=2, covariance=[[1.0, 0.5], [0.5, 1.0]]).dimension
        2
    """

    requires: ClassVar[Requires] = Requires(time={TimeKind.CONTINUOUS})

    dimension: int = 1
    covariance: Any = None

    def declare_handle(self, path: str) -> Expression | SymbolicVector:
        """Return the handle `dW` that noise laws name the processes of the element at a path by.

        Args:
            path:
                The path of the element in its system.

        Returns:
            handle:
                One symbol per process, a single symbol when the element has one process.

        Examples:
            >>> from skcomplex.core.components import Wiener
            >>> Wiener('market', dimension=2).declare_handle('market')
            (dW_market_0, dW_market_1)
        """
        base = f'dW_{re.sub(r"[^0-9A-Za-z]", "_", path)}'
        if self.dimension == 1:
            return symbol(base)
        return vector(symbol(f'{base}_{index}') for index in range(self.dimension))
