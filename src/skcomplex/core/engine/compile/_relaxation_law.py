"""Define the relaxation law the compiler runs for an update law of a relaxation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from typing import ClassVar

from ...components.laws import BaseLaw, RequirementSpec
from ...components.spaces import LatticeSpace, NetworkSpace
from ...components.spaces._discrete_time import _DiscreteTime


class RelaxationLaw(BaseLaw):
    """A law that drives a lattice once per step and relaxes it until it is stable.

    The compiler runs it for an `UpdateLaw` whose value is a call of `relax`. The drive laws run once per step. Then
    the rules run again and again until `stable` holds, which is one cascade. Each cascade that changes the lattice is
    recorded as an event with its size, its duration, and its area.

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
        >>> from skcomplex.core.components.laws import UpdateLaw
        >>> from skcomplex.core.components.spaces import RadiusNeighborhood
        >>> from skcomplex.core.engine.compile._relaxation_law import RelaxationLaw
        >>> height = var('height')
        >>> cross = RadiusNeighborhood(1, metric='manhattan', center=True)
        >>> topple = UpdateLaw(height, lambda z, *n: z - 4 * (z >= 4), neighborhood=cross)
        >>> RelaxationLaw([], [topple], lambda z: bool((z < 4).all())).prepare().targets_
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
        """Return what the relaxation requires: discrete time on a lattice or a network."""
        return RequirementSpec(time=(_DiscreteTime,), space=(LatticeSpace, NetworkSpace))

    def _prepare(self, context: str | None = None) -> None:
        """Check every drive law and every law that relaxes the lattice.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.
        """
        super()._prepare(context)
        for law in (*self.drive, *self.relax):
            law.prepare(context)
