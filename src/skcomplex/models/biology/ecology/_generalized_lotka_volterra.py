"""Interactions of the species of a group, `x_i sum_j a_ij x_j`, through an interaction matrix parameter `a`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.interactions import CouplingInteraction
from ....specs import ParameterSpec, RoleSpec


class GeneralizedLotkaVolterraInteraction(CouplingInteraction):
    """Interactions of the species of a group, `dx_i/dt += x_i sum_j a_ij x_j`, through an interaction matrix `a`.

    Every species `i` couples to every species `j` of the group, itself included, through the term `x_i x_j` weighted
    by the entry `a_ij` of the interaction matrix, a parameter with one row and one column per species, so a schedule or
    an experiment changes it. The diagonal entries are the interactions of a species with itself, negative for
    self-limitation. With the growth `b_i x_i` as a rate, it gives the generalized Lotka-Volterra equations
    `dx_i/dt = x_i (b_i + sum_j a_ij x_j)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of species.

        a:
            The interaction matrix, where `a_ij` is the effect of species `j` on the growth of species `i`, or a number
            for every entry.

        state:
            The name of the state that holds the abundance of a species.

    Examples:
        >>> from skcomplex.models.biology.ecology import GeneralizedLotkaVolterraInteraction
        >>> matrix = ((-1.0, 0.5), (0.5, -1.0))
        >>> GeneralizedLotkaVolterraInteraction('glv', target='species', a=matrix).check().a
        ((-1.0, 0.5), (0.5, -1.0))
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        a: Annotated[object, ParameterSpec(shape=('target', 'target'))] = -1.0,
        state: str = 'x',
    ) -> None:
        super().__init__(
            label,
            target=target,
            read=state,
            write=state,
            function=self._product,
            mode='group',
            weights='a',
            include_self=True,
        )
        self.a = a
        self.state = state

    def _product(self, namespace: SimpleNamespace) -> object:
        """Return the product `x_i x_j` of the abundances of the species and the other species."""
        return getattr(namespace.member, self.state) * getattr(namespace.other, self.state)
