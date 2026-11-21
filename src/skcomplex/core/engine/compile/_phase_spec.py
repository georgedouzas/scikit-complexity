"""Define the spec of a named phase of a step of agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

from ...base import BaseSpec


@dataclass(frozen=True)
class PhaseSpec(BaseSpec):
    """A spec of a named phase of a step, in which agents run some of their rules.

    A step of an agent model can be split into phases, such as producing and then trading, and each rule names its
    phase. In a phase the agents run their rules in the order its activation policy gives: `'random'`, every agent in a
    new random order, `'sequential'`, every agent in the order of creation, `'simultaneous'`, every agent reading the
    states of the start of the phase while its changes apply at its end, `'by-type'`, group after group, each in random
    order, and `'visits'`, visits to agents drawn at random with replacement, `visits` times the number of agents. The
    rules of interactions run once after the agents. A phase repeats `repeat` times per step, such as rounds of trading
    within a day, and runs in the steps `k` for which `k - offset` is a multiple of `period`, so a period of seven runs
    it once a week. Phases order the days of a market, the seasons of a farm, and the rounds of a game.

    Read more in the user guide, under Agents.

    Args:
        name:
            The name of the phase, which rules name in `UpdateLaw(s, rule, phase=name)`.

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

    See Also:
        PhasesScheduleSpec: Runs the phases in order at every step.

    Examples:
        >>> from skcomplex.core.engine.compile import PhaseSpec
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
