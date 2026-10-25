"""Define Wiener elements, sets of Wiener processes with a constant covariance that noise laws share."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar

from ..base import Requires, TimeKind
from ._components import ElementComponent


class Wiener(ElementComponent):
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

        **values:
            The values of the declared variables, by name.

    Examples:
        >>> from skcomplex.core.components import Wiener
        >>> Wiener('market', dimension=2, covariance=[[1.0, 0.5], [0.5, 1.0]]).dimension
        2
    """

    requires: ClassVar[Requires] = Requires(time={TimeKind.CONTINUOUS})

    def __init__(
        self,
        label: str,
        *,
        dimension: int = 1,
        covariance: Any = None,  # noqa: ANN401
        **values: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(label, **values)
        self.dimension = dimension
        self.covariance = covariance
