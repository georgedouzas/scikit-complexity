"""Define the law that drives a lattice and relaxes it until it is stable."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

from ...components.laws import BaseLaw, RequirementSpec
from ...components.spaces import LatticeSpace, NetworkSpace
from ...components.spaces._discrete_time import _DiscreteTime


@dataclass(frozen=True)
class RelaxationLaw(BaseLaw):
    """A law that drives a lattice once per step and relaxes it until it is stable.

    The compiler runs it for an `UpdateLaw` whose value calls `relax`, such as a sandpile that gets one grain per step
    and then topples until no site holds four grains. The drive laws run once per step, then the relaxing rules run
    again and again until `stable` holds, which makes one cascade. Each cascade that changes the lattice is recorded as
    an event with its size, its duration, and its area. It models avalanches of sand, earthquakes along a fault, and
    forest fires.

    Read more in the user guide, under Laws.

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

    See Also:
        ArrayUpdateLaw: Replaces a whole array state by a function of it.
        CascadeKernel: Runs the cascades of a lattice.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.physics.criticality import SandpileSystem
        >>> compilation = PassCompiler().prepare(SandpileSystem()).compilation_
        >>> [type(law).__name__ for _, laws, _ in compilation.collected for law in laws]
        ['RelaxationLaw']
    """

    law_tag: ClassVar[str] = 'update'

    drive: Sequence[BaseLaw]
    relax: Sequence[BaseLaw]
    stable: Callable[..., bool]
    max_iterations: int = 100000
    activity: Callable[..., object] | None = None
    _: KW_ONLY
    label: str = ''

    @classmethod
    def _get_name(cls) -> str:
        """Return the name of the law in messages, the class users write it with."""
        return 'UpdateLaw'

    def _get_targets(self) -> tuple[object, ...]:
        """Return the lattice states the laws that relax the lattice update, each once."""
        return tuple(dict.fromkeys(target for law in self.relax for target in law._get_targets()))

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the relaxation requires: discrete time on a lattice or a network."""
        return RequirementSpec(time=(_DiscreteTime,), space=(LatticeSpace, NetworkSpace))

    def _check(self) -> None:
        """Check every drive law and every law that relaxes the lattice."""
        super()._check()
        for law in (*self.drive, *self.relax):
            law._check()
