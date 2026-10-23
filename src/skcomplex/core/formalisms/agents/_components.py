"""Memories of agent rules: the past values of states that the element types of agents remember."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np
import numpy.typing as npt

type Array = npt.NDArray[np.float64]


def memory(agent: Any, state: str, window: int | None = None) -> Array:  # noqa: ANN401
    """Return the past values of a state of an agent, recorded at the end of every step, the last one last.

    The agent stepper records the states that the element type of the agent lists in its class variable `remembers`,
    which maps each state to the number of values kept.

    Args:
        agent:
            The agent view a rule receives.

        state:
            The name of a remembered state.

        window:
            The number of last values, or `None` for every kept value.

    Returns:
        values:
            The values, an empty array before the first step.

    Raises:
        KeyError:
            The element type does not remember the state.
    """
    mesa_agent = object.__getattribute__(agent, '_agent')
    stepper = object.__getattribute__(agent, '_stepper')
    element = stepper.spec_of(mesa_agent).element
    element_type = getattr(element, 'element', type(element))
    if state not in (getattr(element_type, 'remembers', None) or {}):
        error_msg = f'{element_type.__name__} does not remember {state!r}. List it in the class variable `remembers`.'
        raise KeyError(error_msg)
    values = list(getattr(mesa_agent, 'memories_', {}).get(state, ()))
    values = values if window is None else values[len(values) - min(window, len(values)) :]
    return np.asarray(values, dtype=float)
