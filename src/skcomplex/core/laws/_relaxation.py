"""Define relaxation laws, a slow drive followed by a fast relaxation until a lattice is stable."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from typing import Self

from ...specs import RequirementSpec
from ..spaces import DiscreteTime, LatticeSpace, NetworkSpace
from ._base import BaseLaw


class RelaxationLaw(BaseLaw):
    """Slow drive and fast relaxation: drive laws once per step, then relaxation laws until the lattice is stable.

    Every relaxation that changes the lattice is recorded as an event with its size, the number of site updates, its
    duration, the number of iterations, and its area, the number of distinct sites updated. A site update is a site
    that changes in an iteration, or a site that `activity` marks as active before it, such as a toppling site.

    Args:
        drive:
            The laws applied once per step, local rules or drives.

        relax:
            The laws applied in turn until the lattice is stable.

        stable:
            A function of the arrays of the targets of the relaxation laws, in lattice shape, that returns whether the
            lattice is stable.

        max_iterations:
            The largest number of relaxation iterations in one step.

        activity:
            A function of the arrays of the targets that returns a boolean array of the active sites of an iteration,
            or `None` to count the sites that change.

        label:
            The name of the relaxation events, or the empty string for the path of the element.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws import LocalRuleLaw, RelaxationLaw
        >>> from skcomplex.core.spaces import VonNeumannNeighborhood
        >>> height = var('height')
        >>> topple = LocalRuleLaw(height, VonNeumannNeighborhood(center=True), lambda z, *n: z - 4 * (z >= 4))
        >>> RelaxationLaw([], [topple], lambda z: bool((z < 4).all())).check().prepare().targets_
        (height,)
    """

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
        """Return the lattice states the relaxation laws update, each once."""
        return tuple(dict.fromkeys(target for law in self.relax for target in law._get_targets()))

    def _get_requirements(self) -> RequirementSpec:
        """Return what the relaxation requires of a system: discrete time on a lattice or a network."""
        return RequirementSpec(time=(DiscreteTime,), space=(LatticeSpace, NetworkSpace))

    def _check(self) -> Self:
        """Check every drive and relaxation law.

        Returns:
            law:
                The law itself.
        """
        for law in (*self.drive, *self.relax):
            law.check()
        return self
