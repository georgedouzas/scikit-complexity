"""Relaxation of a density in phase space towards its maximum-entropy equilibrium, the BGK collision model."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

from ....core.interactions import RelaxationInteraction
from ....core.laws.utils import maximum_entropy
from ....specs import RoleSpec


class BhatnagarGrossKrookInteraction(RelaxationInteraction):
    """Relaxation of Bhatnagar, Gross, and Krook of a density towards the maximum-entropy density with its moments.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the field on a phase space.

        state:
            The name of the state of the density.

        axes:
            The names of the velocity axes.

        time:
            A function of the namespace that returns the relaxation time, or a positive number.

    Examples:
        >>> from skcomplex.models.physics.fluids import BhatnagarGrossKrookInteraction
        >>> BhatnagarGrossKrookInteraction('bgk', target='gas', axes=('v',), time=0.1).axes
        ('v',)
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'f',
        axes: Sequence[str] = ('v',),
        time: Callable[[SimpleNamespace], object] | float = 1.0,
    ) -> None:
        super().__init__(label, target=target, state=state, goal=0.0, time=time)
        self.axes = axes

    def _get_goal(self, namespace: SimpleNamespace) -> object:
        """Return the maximum-entropy density with the moments of the state of the target."""
        return maximum_entropy(getattr(namespace.target, self.state), axes=self.axes)
