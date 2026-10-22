"""Define phases, the ordered parts of a step of agents with their own rules and activation policy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

ACTIVATIONS = ('random', 'sequential', 'simultaneous', 'by-type')
DEFAULT_PHASE = 'default'


@dataclass(frozen=True)
class Phase:
    """Named part of a step, in which agents run the rules of the phase under an activation policy.

    The policies are `'random'`, every agent in a new random order, `'sequential'`, every agent in the order of
    creation, `'simultaneous'`, every agent in random order reading the states of the start of the phase while its
    writes apply at the end of the phase, and `'by-type'`, group after group in the order of `groups` or of
    declaration, each group in random order. The rules of interactions and institutions in the phase run once after
    the agents. A phase with `repeat` above one runs that many rounds, such as intraday rounds within a day.

    Args:
        name:
            The name of the phase, which rules name in `Rule(function, phase=name)`.

        activation:
            The activation policy.

        repeat:
            The number of rounds of the phase per step.

        groups:
            The paths of the agent groups in their order of activation for `'by-type'`, or `None` for the order of
            declaration.

    Examples:
        >>> from skcomplex.core.space import Phase
        >>> Phase('trade', activation='simultaneous', repeat=3)
        Phase(name='trade', activation='simultaneous', repeat=3, groups=None)
    """

    name: str
    activation: str = 'random'
    repeat: int = 1
    groups: Sequence[str] | None = None

    def __post_init__(self) -> None:
        """Check the activation policy and the number of rounds."""
        if self.activation not in ACTIVATIONS:
            error_msg = f'Unknown activation {self.activation!r} of phase {self.name!r}. Use one of {ACTIVATIONS}.'
            raise ValueError(error_msg)
        if self.repeat < 1:
            error_msg = f'Phase {self.name!r} repeats {self.repeat} times. Use at least one round.'
            raise ValueError(error_msg)
