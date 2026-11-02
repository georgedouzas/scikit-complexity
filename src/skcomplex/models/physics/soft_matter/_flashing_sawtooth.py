"""Sawtooth potential on overdamped particles on a line, turned on and off at given times."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.interactions import RateInteraction
from ....core.laws.utils import mod, series, where
from ....specs import ParameterSpec, RoleSpec


class FlashingSawtoothInteraction(RateInteraction):
    """Sawtooth potential on overdamped particles on a line, turned on and off at given times.

    Its force is the rate of the position, divided by the friction `gamma` of the target.

    The potential of period `L` has a minimum at every multiple of `L`, rises linearly by `barrier` over the fraction
    `asymmetry` of the period, and falls back over the rest. It is on from time 0 to the first of the `switches`, off
    until the next one, and so on, and it keeps its state after the last one.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or the group.

        barrier:
            The height of the sawtooth.

        L:
            The period of the sawtooth.

        asymmetry:
            The fraction of the period over which the potential rises, 0.5 for a symmetric potential.

        switches:
            The increasing times at which the potential turns off and on in turn.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import FlashingSawtoothInteraction
        >>> FlashingSawtoothInteraction('ratchet', target='beads', switches=(1.0, 2.5)).switches
        (1.0, 2.5)
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        barrier: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        L: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        asymmetry: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        switches: tuple[float, ...] = (),
    ) -> None:
        super().__init__(label, target=target, state='x', divisor='gamma')
        self.barrier = barrier
        self.L = L
        self.asymmetry = asymmetry
        self.switches = switches

    def _get_rate(self, namespace: SimpleNamespace) -> object:
        """Return the force of the sawtooth while it is on.

        Args:
            namespace:
                The namespace of the symbols the rate can use.

        Returns:
            rate:
                The force, before the division by the friction.
        """
        s = namespace
        times = (0.0, *self.switches)
        on = series(s.t, times, [float(index % 2 == 0) for index in range(len(times))], 'previous')
        rise = s.asymmetry * s.L
        return on * where(mod(s.target.x[0], s.L) < rise, -s.barrier / rise, s.barrier / (s.L - rise))
