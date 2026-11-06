"""Metropolis sweeps of Ising spins with quenched couplings to their neighbours, of energy `-s (sum J_k s_k + h)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from ....core.components.laws.utils import neighbors
from ....core.components.spaces import VonNeumannNeighborhood
from ....core.specs import RoleSpec
from ..common.spins import SpinSamplingInteraction

NEAREST = VonNeumannNeighborhood()


class BondSamplingInteraction(SpinSamplingInteraction):
    """Metropolis sweeps of Ising spins with quenched couplings to their neighbours, of energy `-s (sum J_k s_k + h)`.

    The couplings `J_up`, `J_left`, `J_right`, and `J_down` are constant states of every spin, in the order of the
    offsets of the von Neumann neighbourhood, and the field `h` and the temperature are its parameters.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the lattice group of spins.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import BondSamplingInteraction
        >>> BondSamplingInteraction('sampling', target='spins').neighborhood
        VonNeumannNeighborhood()
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, state='spin', energy=self._energy, neighborhood=NEAREST)

    def _energy(self, namespace: SimpleNamespace) -> Expression:
        """Return the local energy `-s (sum_k J_k s_k + h)` of a spin."""
        s = namespace.target
        couplings = (s.J_up, s.J_left, s.J_right, s.J_down)
        neighbours = neighbors(s.spin, NEAREST, reduce='each')
        return -s.spin * (sum(coupling * other for coupling, other in zip(couplings, neighbours, strict=True)) + s.h)
