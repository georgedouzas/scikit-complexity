"""Define contour uptake, which mixes the logistic map of every cell with the power uptake from the next cell."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.interactions import CouplingInteraction
from ....core.types import Target
from ....specs import RoleSpec


class ContourUptakeInteraction(CouplingInteraction):
    """Uptake of every cell of a contour from the next cell, `x_i <- (1 - c_i) r x_i (1 - x_i) + c_i x_{i+1}^p_i`.

    It is the map form of coupling to the next member of the group, with the power `x_{i+1}^{p_i}` of the next cell
    combined with the logistic map of the cell. The last cell of the contour takes up from the first.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of cells.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.core.spaces import DiscreteTime
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.biology.physiology import ContourCellElement, ContourUptakeInteraction
        >>> cells = GroupElement('cells', element=ContourCellElement, size=2, values={'x': [0.25, 1.0], 'c': 0.5})
        >>> uptake = ContourUptakeInteraction('uptake', target='cells')
        >>> system = System(elements=[cells], interactions=[uptake], time=DiscreteTime())
        >>> Simulation(system).prepare().run(until=1).trajectory_.values('cells.x')[-1].round(4).tolist()
        [0.7344, 0.25]
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(
            label,
            target=target,
            read='x',
            write='x',
            function=self._get_power,
            mode='next',
            form='map',
            combine=self._get_mix,
        )

    @staticmethod
    def _get_power(namespace: SimpleNamespace) -> Target:
        """Return the power uptake of the next cell, with the affinity of the cell."""
        return namespace.other.x**namespace.member.p

    @staticmethod
    def _get_mix(namespace: SimpleNamespace, reduction: Target) -> Target:
        """Return the logistic map of the cell mixed with the uptake from the next cell."""
        member = namespace.member
        return (1 - member.c) * member.r * member.x * (1 - member.x) + member.c * reduction
