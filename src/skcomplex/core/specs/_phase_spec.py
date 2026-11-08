"""Define the phase spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class PhaseSpec(BaseSpec):
    """Named part of a step.

    In it agents run the rules of the phase under an activation policy. The policies are `'random'`, every agent in a
    new random order, `'sequential'`, every agent in the order of creation, `'simultaneous'`, every agent in random
    order reading the states of the start of the phase while its writes apply at the end of the phase, `'by-type'`,
    group after group in the order of `groups` or of declaration, each group in random order, and `'visits'`, as many
    visits as `visits` times the number of agents of `groups` at the start of the phase, each to an agent drawn at
    random with replacement, which runs its rules of the phase at the time `k + i / n` of visit `i` of `n` in step `k`.
    The rules of interactions and institutions in the phase run once after the agents. A phase with `repeat` above one
    runs that many rounds, such as intraday rounds within a day. A phase runs in the steps `k`, counted from one, for
    which `k - offset` is a multiple of `period`, so the default period of one runs it every step and a period of seven
    runs it once a week.

    Args:
        name:
            The name of the phase, which rules name in `RuleLaw(function, phase=name)`.

        activation:
            The activation policy.

        repeat:
            The number of rounds of the phase per step.

        groups:
            The paths of the agent groups in their order of activation for `'by-type'`, or the groups whose agents
            `'visits'` draws, or `None` for every group in the order of declaration.

        visits:
            The number of visits of `'visits'` per agent of the groups at the start of the phase.

        period:
            The number of steps between two runs of the phase.

        offset:
            The step, modulo the period, in which the phase runs.

    Examples:
        >>> from skcomplex.core.specs import PhaseSpec
        >>> PhaseSpec('trade', activation='simultaneous', repeat=3)
        PhaseSpec(name='trade', activation='simultaneous', repeat=3, groups=None, visits=1.0, period=1, offset=0)
    """

    name: str
    activation: str = 'random'
    repeat: int = 1
    groups: Sequence[str] | None = None
    visits: float = 1.0
    period: int = 1
    offset: int = 0
