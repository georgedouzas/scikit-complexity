"""Define the law that drives a lattice and relaxes it until it is stable."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from typing import ClassVar

from ...specs import RequirementSpec
from ..spaces import DiscreteTime, LatticeSpace, NetworkSpace
from ._base import BaseLaw


class CascadeLaw(BaseLaw):
    """A law that drives a lattice once per step and relaxes it until it is stable.

    The drive laws run once per step. Then the relax laws run again and again until `stable` holds, which is one
    cascade, or avalanche. Each cascade that changes the lattice is recorded as an event with its size, the number of
    site updates, its duration, the number of iterations, and its area, the number of distinct sites updated. A site
    update is a site that changes in an iteration, or a site that `activity` marks as active before it, such as a
    toppling site.

    Unlike `UpdateLaw`, which runs once per step, it runs its relax laws until the lattice settles, so one step holds a
    whole avalanche. Choose it when a slow drive and fast relaxation are separated in time, as in self-organized
    criticality.

    It appears in many fields. Grains topple in the sandpile model of Bak, Tang, and Wiesenfeld, slips spread along a
    fault in the earthquake model of Olami, Feder, and Christensen, and trees burn in a forest fire model.

    Args:
        drive:
            The laws applied once per step, local rules or drives.

        relax:
            The laws applied in turn until the lattice is stable.

        stable:
            A function of the arrays of the targets of the laws that relax the lattice, in lattice shape,
            that returns whether the lattice is stable.

        max_iterations:
            The largest number of iterations in one step.

        activity:
            A function of the arrays of the targets that returns a boolean array of the active sites of an iteration,
            or `None` to count the sites that change.

        label:
            The name of the cascade events, or the empty string for the path of the element.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import CascadeLaw, LocalRuleLaw
        >>> from skcomplex.core.components.spaces import BallNeighborhood
        >>> height = var('height')
        >>> cross = BallNeighborhood(metric='manhattan', center='first')
        >>> topple = LocalRuleLaw(height, cross, lambda z, *n: z - 4 * (z >= 4))
        >>> CascadeLaw([], [topple], lambda z: bool((z < 4).all())).prepare().targets_
        (height,)
    """

    law_tag: ClassVar[str] = 'update'

    def __init__(
        self,
        drive: Sequence[BaseLaw],
        relax: Sequence[BaseLaw],
        stable: Callable[..., bool],
        max_iterations: int = 100_000,
        activity: Callable[..., object] | None = None,
        *,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.drive = drive
        self.relax = relax
        self.stable = stable
        self.max_iterations = max_iterations
        self.activity = activity

    def _get_targets(self) -> tuple[object, ...]:
        """Return the lattice states the laws that relax the lattice update, each once."""
        return tuple(dict.fromkeys(target for law in self.relax for target in law._get_targets()))

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the cascade requires: discrete time on a lattice or a network."""
        return RequirementSpec(time=(DiscreteTime,), space=(LatticeSpace, NetworkSpace))

    def _prepare(self, context: str | None = None) -> None:
        """Check every drive law and every law that relaxes the lattice.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.
        """
        super()._prepare(context)
        for law in (*self.drive, *self.relax):
            law.prepare(context)
