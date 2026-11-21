"""Define the transfers of agent rules."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....errors import ValidationError


def transfer(source: object, sink: object, amount: float, state: str, sink_state: str | None = None) -> None:
    """Move an amount of a quantity from one agent to another inside a rule.

    It subtracts `amount` from the value `state` of `source` and adds it to the value of `sink`, so the total over all
    agents stays the same, such as 4 units moving from a holding of 10 to a holding of 0. It is what a rule of an
    `UpdateLaw` calls for a flow that `RateLaw({s.source.cash: -1, s.target.cash: +1}, f)` states for a whole group. A
    negative holding is an amount the agent owes. It pays a seller, hands food from one ant to another, and moves charge
    between two cells.

    Read more in the user guide, under Agents.

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

    See Also:
        UpdateLaw: Runs the rule that calls it.

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
