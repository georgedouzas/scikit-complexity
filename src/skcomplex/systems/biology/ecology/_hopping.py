"""Hops of single organisms between neighbouring patches, each organism moving to each neighbour at the rate `beta`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, JumpLaw
from ....core.specs import ParameterSpec, RoleSpec


class HoppingInteraction(Interaction):
    """Hops of single organisms between neighbouring patches, each organism moving to each neighbour at the rate `beta`.

    The patches are the entries of a variadic role, and every pair `(i, j)` of `links` gives a jump that moves one
    organism from the patch of entry `i` to the patch of entry `j` at the rate `beta n_i`.

    Args:
        label:
            The label of the interaction.

        patches:
            The paths of the patches, such as the members `patches[i]` of a lattice.

        beta:
            The rate at which an organism hops to one neighbouring patch.

        links:
            The ordered pairs of entries of `patches` between which organisms hop.

    Examples:
        >>> from skcomplex.systems.biology.ecology import HoppingInteraction
        >>> HoppingInteraction('hopping', patches=['a', 'b'], links=[(0, 1), (1, 0)]).beta
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        patches: Annotated[list[str], RoleSpec(variadic=True)],
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        links: Sequence[tuple[int, int]] = (),
    ) -> None:
        super().__init__(label)
        self.patches = patches
        self.beta = beta
        self.links = links

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return one hop per ordered pair of neighbouring patches.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        patches = s.patches
        return [JumpLaw(s.beta * patches[i].n, {patches[i].n: -1, patches[j].n: 1}) for i, j in self.links]
