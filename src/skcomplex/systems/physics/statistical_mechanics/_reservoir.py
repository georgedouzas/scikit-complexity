"""Reservoir that adds one particle to each of its sites at rate `inflow` and removes one at rate `outflow`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import TransferInteraction
from ....core.components.laws import Source
from ....core.components.laws.utils import where
from ....core.specs import ParameterSpec, RoleSpec


class ReservoirInteraction(TransferInteraction):
    """Reservoir that adds one particle to each of its sites at rate `inflow` and removes one at rate `outflow`.

    A particle leaves only an occupied site. With exclusion a particle enters only an empty site. It is a transfer of
    one particle from the environment into every site, with jumps back to the environment at the outflow rate.

    Args:
        label:
            The label of the reservoir.

        sites:
            The sites in contact with the reservoir.

        inflow:
            The rate at which a particle enters a site.

        outflow:
            The rate at which a particle leaves an occupied site.

        exclusion:
            Whether a site holds at most one particle.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ReservoirInteraction
        >>> ReservoirInteraction('bath', sites=['lane[0]'], inflow=0.5).outflow
        0.0
    """

    def __init__(
        self,
        label: str,
        *,
        sites: Annotated[list[str], RoleSpec(variadic=True)],
        inflow: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        outflow: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        exclusion: bool = False,
    ) -> None:
        super().__init__(
            label,
            source='',
            target='',
            state=Source(),
            into='n',
            amount=1.0,
            rate=self._get_entry_rate,
            reverse_rate=self._get_exit_rate,
        )
        self.sites = sites
        self.inflow = inflow
        self.outflow = outflow
        self.exclusion = exclusion

    def _get_holders(self, namespace: SimpleNamespace) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
        """Return every site as the target of its own entries."""
        return [(site, site) for site in namespace.sites]

    def _get_entry_rate(self, s: SimpleNamespace) -> Expression:
        """Return the rate at which a particle enters a site, only an empty one under exclusion."""
        return s.inflow * (1 - s.target.n if self.exclusion else 1)

    @staticmethod
    def _get_exit_rate(s: SimpleNamespace) -> Expression:
        """Return the rate at which a particle leaves an occupied site."""
        return s.outflow * where(s.target.n > 0, 1, 0)
