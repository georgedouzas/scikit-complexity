"""Define the readout of a simple recurrent network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp

from ....core.interactions import CouplingInteraction
from ....specs import ParameterSpec, RoleSpec


class ReadoutCouplingInteraction(CouplingInteraction):
    """Readout of a simple recurrent network, `y_i <- sigmoid(sum_j W_ij s_j + c_i)` with the logistic sigmoid.

    Every output unit reads the activities of the hidden units through the adaptive matrix `W`, adds its bias `c`, and
    takes the sigmoid of the sum as its next prediction.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of output units.

        source:
            The path of the group of hidden units.

        W:
            The readout weights, a matrix of shape `(output units, hidden units)` or a number for every entry.

    Examples:
        >>> from skcomplex.models.computer_science.neural_networks import ReadoutCouplingInteraction
        >>> ReadoutCouplingInteraction('readout', target='outputs', source='hidden').write
        'y'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
        W: Annotated[object, ParameterSpec(shape=('target', 'source'), adaptive=True)] = 0.0,
    ) -> None:
        super().__init__(
            label,
            target=target,
            source=source,
            read='s',
            write='y',
            function=self._activity,
            mode='group',
            weights='W',
            form='map',
            phase='readout',
            combine=self._activate,
        )
        self.W = W

    def _activity(self, namespace: SimpleNamespace) -> object:
        """Return the activity of the hidden unit."""
        return namespace.other.s

    def _activate(self, namespace: SimpleNamespace, reduction: object) -> object:
        """Return the sigmoid of the weighted activities and the bias of the output unit."""
        return 1 / (1 + exp(-(reduction + namespace.member.c)))
