"""Memories of agent rules: the past values of the states of agents that keep lags."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt

from ..problems import VariableKind, declarations_of
from ._mesa import AgentView, _agent_type

type Array = npt.NDArray[np.float64]


def memory(agent: AgentView, state: str, window: int | None = None) -> Array:
    """Return the past values of a state of an agent, the values at the ends of the last steps, the last one last.

    An agent remembers a state declared with lags, `StateSpec(lags=m)`: the agent stepper keeps its last `m` values
    in the companion states `<state>_lag1` to `<state>_lag<m>`, which rules may also read one by one. Before the
    first steps the lags hold their initial values, the `initial` of the spec or else the initial value of the state.

    Args:
        agent:
            The agent view a rule receives.

        state:
            The name of a state with lags.

        window:
            The number of last values, or `None` for every kept value.

    Returns:
        values:
            The values, the oldest first.

    Raises:
        KeyError:
            The state of the agent keeps no lags.
    """
    spec = agent._stepper._spec_of(agent._agent)
    declaration = declarations_of(_agent_type(spec), VariableKind.STATE).get(state)
    if declaration is None or not declaration.lags:
        error_msg = (
            f'The agents of {spec.path} do not remember {state!r}. Declare the state with lags, such as '
            '`StateSpec(lags=3)`, to keep its last values.'
        )
        raise KeyError(error_msg)
    lags = declaration.lags if window is None else min(window, declaration.lags)
    return np.asarray([getattr(agent, f'{state}_lag{index}') for index in range(lags, 0, -1)], dtype=float)
