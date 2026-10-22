"""Learning blocks of agent rules: imitation of the best-scoring neighbour."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from typing import Any

type Score = str | Callable[[Any], float]


def _scorer(score: Score) -> Callable[[Any], float]:
    """Return a function of an agent from a state name or a function."""
    if isinstance(score, str):
        return lambda agent: float(getattr(agent, score))
    return score


def imitate(agent: Any, neighbors: Sequence[Any], score: Score, states: str | Sequence[str]) -> Any:  # noqa: ANN401
    """Copy states of the neighbour with the best score when it scores better than the agent.

    In a simultaneous phase the agent reads its neighbours' states from the start of the phase.

    Args:
        agent:
            The agent view.

        neighbors:
            The neighbouring agent views.

        score:
            The state name or the function of an agent that scores it, such as its payoff.

        states:
            The name or names of the states copied, such as the strategy.

    Returns:
        model:
            The imitated neighbour, or the agent itself when no neighbour scores better.
    """
    scorer = _scorer(score)
    best, best_score = agent, scorer(agent)
    for neighbor in neighbors:
        value = scorer(neighbor)
        if value > best_score:
            best, best_score = neighbor, value
    if best is not agent:
        for name in [states] if isinstance(states, str) else states:
            setattr(agent, name, getattr(best, name))
    return best
