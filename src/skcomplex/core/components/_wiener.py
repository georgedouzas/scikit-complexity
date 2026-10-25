"""Define Wiener processes, sets of processes with a constant covariance that the noise laws of interactions draw."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar

from ..base import BaseEstimator, Requires, TimeKind


class WienerProcess(BaseEstimator):
    """Set of Wiener processes whose increments over a step `h` have the covariance `h C`.

    The processes hold no states and are no component. An interaction binds them through a role declared with
    `role(noise=True)`, and its noise laws name them through the handle `s.<role>.dW`, a symbol for one process and a
    vector of symbols for several, so `s.<role>.dW[k]` names process `k`. Interactions of one system that bind
    processes of one label share them, so laws of any component that name one process share its increment, and group
    laws that name it give all members one common process. The covariance is fixed when the system is compiled.

    Args:
        label:
            The label of the processes, unique among the children of the system of the interactions that bind them.

        dimension:
            The number of processes.

        covariance:
            The constant covariance matrix `C` of the processes, symmetric and positive semidefinite of size
            `dimension`, or `None` for the identity.

    Examples:
        >>> from skcomplex.core.components import WienerProcess
        >>> WienerProcess('market', dimension=2, covariance=[[1.0, 0.5], [0.5, 1.0]]).dimension
        2
    """

    requires: ClassVar[Requires] = Requires(time={TimeKind.CONTINUOUS})

    def __init__(
        self,
        label: str,
        *,
        dimension: int = 1,
        covariance: Any = None,  # noqa: ANN401
    ) -> None:
        self.label = label
        self.dimension = dimension
        self.covariance = covariance
