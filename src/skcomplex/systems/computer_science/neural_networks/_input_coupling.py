"""Define the input weights of a simple recurrent network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.interactions import CouplingInteraction
from ....core.specs import ParameterSpec, RoleSpec


class InputCouplingInteraction(CouplingInteraction):
    """Weighted input of a simple recurrent network, which sets the drive `u = D x` of every hidden unit each step.

    The weights `D` are an adaptive matrix with one row per hidden unit and one column per input unit, so training
    changes them. The coupling runs in the phase `drive` of the step, which the system declares before the phases
    `hidden` of the recurrence and `readout` of the readout, so the hidden units read the drive of the same step.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of hidden units.

        source:
            The path of the group of input units.

        D:
            The input weights, a matrix of shape `(hidden units, input units)` or a number for every entry.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import InputCouplingInteraction
        >>> InputCouplingInteraction('input', target='hidden', source='inputs').write
        'u'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
        D: Annotated[object, ParameterSpec(shape=('target', 'source'), adaptive=True)] = 0.0,
    ) -> None:
        super().__init__(
            label,
            target=target,
            source=source,
            read='x',
            write='u',
            function=self._input,
            mode='group',
            weights='D',
            form='map',
            phase='drive',
        )
        self.D = D

    def _input(self, namespace: SimpleNamespace) -> object:
        """Return the input of the other unit."""
        return namespace.other.x
