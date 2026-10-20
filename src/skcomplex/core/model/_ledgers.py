"""Define the ledgers of agents: atomic transfers of quantities and balance-sheet identities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import pickle  # nosec B403: pickles the identity's function to check that it pickles, and never loads
from collections.abc import Callable
from typing import Any

from ..base import ConservationError, ValidationError


def transfer(source: Any, sink: Any, amount: float, state: str, sink_state: str | None = None) -> None:  # noqa: ANN401
    """Move an amount of a quantity from one agent to another at once, inside a rule.

    Both holdings change together, so the total of the quantity over all agents is unchanged. A holder of a negative
    amount owes it, so a debt of 10 is a holding of -10 and a loan is a transfer of claims in the opposite direction.

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
        >>> from skcomplex.core.model import transfer
        >>> household, firm = SimpleNamespace(money=10.0), SimpleNamespace(money=0.0)
        >>> transfer(household, firm, 4.0, 'money')
        >>> household.money, firm.money
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


class Identity:
    """Balance-sheet identity of an agent type, an expression of an agent's states that must stay zero.

    Agent types list their identities in the class variable `identities`, and the agent-based stepper checks every
    identity of every agent after every step.

    Args:
        label:
            The name of the identity, such as `'assets == liabilities + equity'`.

        residual:
            A module-level function of an agent that returns the side difference, such as
            `assets - liabilities - equity`.

        tolerance:
            The largest absolute residual that counts as zero.

    Raises:
        ValidationError:
            The residual cannot be pickled, such as a lambda.

    Examples:
        >>> from types import SimpleNamespace
        >>> from operator import attrgetter
        >>> from skcomplex.core.model import Identity
        >>> identity = Identity('no equity', attrgetter('equity'))
        >>> identity.residual(SimpleNamespace(equity=0.0))
        0.0
    """

    def __init__(self, label: str, residual: Callable[[Any], float], *, tolerance: float = 1e-9) -> None:
        self.label = label
        self.residual = residual
        self.tolerance = tolerance
        try:
            pickle.dumps(residual)
        except (pickle.PicklingError, AttributeError, TypeError) as error:
            error_msg = (
                f'The residual of the identity {label!r} cannot be pickled: {error}. Use a module-level function.'
            )
            raise ValidationError(error_msg) from error

    def __repr__(self) -> str:
        """Return the label of the identity."""
        return f'Identity({self.label!r})'

    def check(self, agent: Any, name: str) -> None:  # noqa: ANN401
        """Raise an error when the identity does not hold for an agent.

        Args:
            agent:
                The agent, as rules receive it.

            name:
                The name of the agent in the message, such as its group and id.

        Raises:
            ConservationError:
                The residual exceeds the tolerance.
        """
        difference = float(self.residual(agent))
        if abs(difference) > self.tolerance:
            error_msg = (
                f'The identity {self.label!r} of {name} does not hold: the difference is {difference:.6g}. Change '
                'balances with transfers, or change both sides of the identity in the same rule.'
            )
            raise ConservationError(error_msg)
