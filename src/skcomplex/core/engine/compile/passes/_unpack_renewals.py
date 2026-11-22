"""Replace every jump that waits on a clock by its renewal law."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....components.laws import JumpLaw
from .._base import BasePass
from .._compilation import Compilation
from .._renewal_law import RenewalLaw


class UnpackRenewalsPass(BasePass):
    """A pass that replaces every jump that waits on a clock by its renewal law.

    A jump with a random waiting time, such as a walker that rests and then jumps, runs on a clock of its own, and the
    pass writes it as a renewal law, which the solver runs.

    Read more in the user guide, under Extend the library.

    See Also:
        RenewalLaw: Renews states after random waiting times.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.physics.soft_matter import ContinuousTimeRandomWalkSystem
        >>> compilation = PassCompiler().prepare(ContinuousTimeRandomWalkSystem()).compilation_
        >>> [type(law).__name__ for _, laws, _ in compilation.collected for law in laws]
        ['RenewalLaw']
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('collected',)

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('collected',)

    def _run(self, inputs: object = None) -> None:
        """Replace the jumps on clocks of waiting times by renewals."""
        compilation = self.compilation_
        compilation.collected = [
            (
                spec,
                [
                    (
                        RenewalLaw(law.waiting, law.reset, label=law.label)
                        if isinstance(law, JumpLaw) and law.waiting is not None
                        else law
                    )
                    for law in laws
                ],
                block,
            )
            for spec, laws, block in Compilation._get_required(compilation.collected, 'collected')
        ]
