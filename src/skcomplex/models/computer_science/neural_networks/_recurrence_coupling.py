"""Define the recurrent weights of a simple recurrent network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp

from ....core.interactions import CouplingInteraction
from ....specs import ParameterSpec, RoleSpec


class RecurrenceCouplingInteraction(CouplingInteraction):
    """Recurrence of a simple recurrent network, `s_i <- sigmoid(sum_j A_ij s_j + u_i + b_i)` with the logistic sigmoid.

    Every hidden unit reads the activities of every hidden unit, itself included, through the adaptive matrix `A`, adds
    its drive `u` and its bias `b`, and takes the sigmoid of the sum as its next activity (Gros 2015 Eq. 10.18).

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of hidden units.

        A:
            The recurrent weights, a square matrix with one row and one column per hidden unit, or a number for every
            entry.

    Examples:
        >>> from skcomplex.models.computer_science.neural_networks import RecurrenceCouplingInteraction
        >>> RecurrenceCouplingInteraction('recurrence', target='hidden').include_self
        True
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        A: Annotated[object, ParameterSpec(shape=('target', 'target'), adaptive=True)] = 0.0,
    ) -> None:
        super().__init__(
            label,
            target=target,
            read='s',
            write='s',
            function=self._activity,
            mode='group',
            weights='A',
            include_self=True,
            form='map',
            phase='hidden',
            combine=self._activate,
        )
        self.A = A

    def _activity(self, namespace: SimpleNamespace) -> object:
        """Return the activity of the other unit."""
        return namespace.other.s

    def _activate(self, namespace: SimpleNamespace, reduction: object) -> object:
        """Return the sigmoid of the recurrent input, the drive, and the bias of the unit."""
        return 1 / (1 + exp(-(reduction + namespace.member.u + namespace.member.b)))
