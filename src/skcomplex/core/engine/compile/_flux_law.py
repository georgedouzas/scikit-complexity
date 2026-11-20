"""Define the flux law the compiler reads from a conservation law."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import ClassVar

from ...components.laws import BaseLaw, RequirementSpec
from ...components.spaces import CellComplexSpace, LatticeSpace
from ...components.spaces._continuous_time import _ContinuousTime
from ...components.types import Target


class _FluxLaw(BaseLaw):
    """A flux law the compiler reads from a conservation law of a field.

    A rate `-divergence(flux)` of a state of a finite-volume field becomes this law, which carries the flux of every
    dimension and the scheme of the field, so the finite-volume kernel computes the differences of the flux between
    cells.

    Args:
        target:
            The symbol of the conserved state.

        flux:
            The flux of every dimension.

        limiter:
            The slope limiter of the field.

        riemann:
            The numerical flux of the field.

        speed:
            The wave speed of the Rusanov flux, or `None`.

        label:
            The name of the rate law the flux comes from.
    """

    stated_as: ClassVar[str] = 'RateLaw'

    law_tag: ClassVar[str] = 'rate'

    def __init__(
        self,
        target: Target,
        flux: Sequence[object],
        *,
        limiter: str | None = 'minmod',
        riemann: str = 'rusanov',
        speed: object = None,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.flux = flux
        self.limiter = limiter
        self.riemann = riemann
        self.speed = speed

    def _get_targets(self) -> tuple[object, ...]:
        """Return the conserved state."""
        return (self.target,)

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires: continuous time on a lattice or a mesh."""
        return RequirementSpec(time=(_ContinuousTime,), space=(LatticeSpace, CellComplexSpace))

    def _get_value(self, inputs: object) -> object:
        """Return the flux at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        return [self._evaluate(item, inputs) for item in self.flux]
