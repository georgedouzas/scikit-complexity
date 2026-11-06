"""Define the helpers of agent rules: transfers between agents, and the pairings of match laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np

from ....errors import ValidationError
from ...types import SystemViewProtocol


def transfer(source: object, sink: object, amount: float, state: str, sink_state: str | None = None) -> None:
    """Move an amount of a quantity from one agent to another at once, inside a rule.

    It is the imperative form of `TransferLaw` for agent rules. Both holdings change together, so the total of the
    quantity over all agents is unchanged, and a negative holding is an amount the agent owes.

    Args:
        source:
            The agent the amount leaves, as the rule receives it.

        sink:
            The agent the amount enters.

        amount:
            The amount moved, negative to move it the other way.

        state:
            The state of the source that holds the quantity.

        sink_state:
            The state of the sink that holds the quantity, or `None` for `state`.

    Raises:
        ValidationError:
            An agent has no such state.

    Examples:
        >>> from types import SimpleNamespace
        >>> from skcomplex.core.components.laws.utils import transfer
        >>> first, second = SimpleNamespace(level=10.0), SimpleNamespace(level=0.0)
        >>> transfer(first, second, 4.0, 'level')
        >>> first.level, second.level
        (6.0, 4.0)
    """
    target = state if sink_state is None else sink_state
    for agent, name in ((source, state), (sink, target)):
        if not hasattr(agent, name):
            error_msg = f'The transfer of {amount} reads the state {name!r}, which the agent {agent!r} does not have.'
            raise ValidationError(error_msg)
    if source is sink and state == target:
        return
    source_value, sink_value = getattr(source, state), getattr(sink, target)
    setattr(source, state, source_value - amount)
    setattr(sink, target, sink_value + amount)


def random_pairing(
    left: Sequence[object],
    right: Sequence[object],
    rng: np.random.Generator,
    capacity: int,
) -> list[tuple[object, object]]:
    """Return pairs drawn uniformly at random, each member in at most `capacity` pairs and never paired with itself.

    Args:
        left:
            The members of the first group.

        right:
            The members of the second group, the same sequence as `left` for pairs within one group.

        rng:
            The random generator.

        capacity:
            The largest number of pairs of every member.

    Returns:
        pairs:
            The pairs of members.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.components.laws.utils import random_pairing
        >>> pairs = random_pairing(list(range(5)), list('abc'), np.random.default_rng(0), 1)
        >>> len(pairs)
        3
    """
    pairs: list[tuple[object, object]] = []
    seen: set[tuple[int, int]] = set()
    if left is right:
        for _ in range(capacity):
            order = rng.permutation(len(left))
            for first, second in zip(order[::2], order[1::2], strict=False):
                key = (min(first, second), max(first, second))
                if key not in seen:
                    seen.add(key)
                    pairs.append((left[first], right[second]))
        return pairs
    firsts = rng.permutation(np.repeat(np.arange(len(left)), capacity))
    seconds = rng.permutation(np.repeat(np.arange(len(right)), capacity))
    for first, second in zip(firsts, seconds, strict=False):
        if (first, second) not in seen:
            seen.add((first, second))
            pairs.append((left[first], right[second]))
    return pairs


def matches(system: SystemViewProtocol, label: str = 'match') -> list[tuple[object, object]]:
    """Return the pairs a matching law stored for the current step.

    Args:
        system:
            The system view a rule receives.

        label:
            The label of the matching law.

    Returns:
        pairs:
            The pairs of agents, or none before the law runs.
    """
    pairs = system.storage.get(f'match:{label}', ())
    return list(pairs) if isinstance(pairs, list | tuple) else []
