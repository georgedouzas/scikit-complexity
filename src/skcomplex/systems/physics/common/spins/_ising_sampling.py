"""Define Ising sampling.

Monte Carlo sweeps flip up and down spins coupled to their neighbours and a field.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from .....core.components.laws.utils import neighbors
from .....core.types import Role
from ._spin_sampling import SpinSamplingInteraction


class IsingSamplingInteraction(SpinSamplingInteraction):
    """Boltzmann weight of Ising spins `s = ±1`, whose local energy is `-J s sum_j s_j - h s`.

    The coupling `J`, the field `h`, and the temperature are values of the spins, and the `MetropolisSolver` of the
    system flips them, `MetropolisSolver(moves={'spins.spin': 'flip'})`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of spins.

        state:
            The name of the state that holds the spin.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.systems.physics.common.spins import IsingSamplingInteraction
        >>> ising = IsingSamplingInteraction('ising', target='spins')
        >>> other = SimpleNamespace(spin=var('t'))
        >>> ising.energy(SimpleNamespace(target=SimpleNamespace(spin=var('s'), J=1, h=var('h'), other=other)))
        -h*s - s*neighbors(t, reduce='sum')
    """

    def __init__(self, label: str, *, target: Role, state: str = 'spin') -> None:
        super().__init__(label, target=target, state=state, energy=self._get_energy)

    def _get_energy(self, namespace: SimpleNamespace) -> Expression:
        """Return the local energy `-J s sum_j s_j - h s` of a spin."""
        spin, other = getattr(namespace.target, self.state), getattr(namespace.target.other, self.state)
        return -namespace.target.J * spin * neighbors(other) - namespace.target.h * spin
