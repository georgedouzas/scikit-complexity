"""Learning blocks of agent rules: memories, forecasting heuristics, logit choice, imitation, and evolution."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import softmax

type Array = npt.NDArray[np.float64]
type Score = str | Callable[[Any], float]

SELECTIONS = ('tournament', 'roulette', 'truncation')


def memory(agent: Any, state: str, window: int | None = None) -> Array:  # noqa: ANN401
    """Return the past values of a state of an agent, recorded at the end of every step, the last one last.

    The stepper records the states that the agent type lists in its class variable `remembers`, which maps each state
    to the number of values kept.

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
            The agent type does not remember the state.
    """
    mesa_agent = object.__getattribute__(agent, '_agent')
    stepper = object.__getattribute__(agent, '_stepper')
    element = stepper.spec_of(mesa_agent).element
    agent_type = getattr(element, 'element', type(element))
    if state not in (getattr(agent_type, 'remembers', None) or {}):
        error_msg = f'{agent_type.__name__} does not remember {state!r}. List it in the class variable `remembers`.'
        raise KeyError(error_msg)
    values = list(getattr(mesa_agent, 'memories_', {}).get(state, ()))
    values = values if window is None else values[len(values) - min(window, len(values)) :]
    return np.asarray(values, dtype=float)


def _adaptive(history: Sequence[float], previous: float, weight: float = 0.65) -> float:
    """Return the adaptive forecast `w x(t) + (1 - w) f(t)` of the last value and the previous forecast."""
    return float(weight * history[-1] + (1 - weight) * previous)


def _trend(history: Sequence[float], strength: float = 1.3) -> float:
    """Return the trend-following forecast `x(t) + g (x(t) - x(t - 1))`."""
    if len(history) < 2:  # noqa: PLR2004
        return float(history[-1])
    return float(history[-1] + strength * (history[-1] - history[-2]))


def _anchoring(history: Sequence[float]) -> float:
    """Return the anchoring-and-adjustment forecast `(mean + x(t)) / 2 + (x(t) - x(t - 1))`."""
    if len(history) < 2:  # noqa: PLR2004
        return float(history[-1])
    return float((np.mean(history) + history[-1]) / 2 + history[-1] - history[-2])


def _fundamental(history: Sequence[float], fundamental: float = 0.0, reversion: float = 0.0) -> float:
    """Return the fundamentalist forecast `F + v (x(t) - F)`."""
    return float(fundamental + reversion * (history[-1] - fundamental))


heuristics = SimpleNamespace(adaptive=_adaptive, trend=_trend, anchoring=_anchoring, fundamental=_fundamental)
heuristics.__doc__ = """Forecasting heuristics of Anufriev and Hommes, functions of the past values, the last one last.

- `adaptive(history, previous, weight)` gives `w x(t) + (1 - w) f(t)`, with `f(t)` the previous forecast.
- `trend(history, strength)` gives `x(t) + g (x(t) - x(t - 1))`.
- `anchoring(history)` gives `(mean + x(t)) / 2 + (x(t) - x(t - 1))`.
- `fundamental(history, fundamental, reversion)` gives `F + v (x(t) - F)`.

Examples:
    >>> from skcomplex.core.model import heuristics
    >>> heuristics.trend([1.0, 2.0], strength=0.5)
    2.5
"""


def logit_probabilities(utilities: npt.ArrayLike, intensity: float) -> Array:
    """Return the logit choice probabilities `exp(beta U_h) / sum_k exp(beta U_k)`.

    Args:
        utilities:
            The utility of every option, such as past performance of every strategy.

        intensity:
            The intensity of choice `beta`, zero for uniform choice and infinite for the best option.

    Returns:
        probabilities:
            One probability per option.

    Examples:
        >>> from skcomplex.core.model import logit_probabilities
        >>> logit_probabilities([0.0, 0.0], 1.0).tolist()
        [0.5, 0.5]
    """
    values = np.asarray(utilities, dtype=float)
    if np.isinf(intensity):
        best = values == values.max()
        return best / best.sum()
    return np.asarray(softmax(intensity * values), dtype=float)


def logit_choice(utilities: npt.ArrayLike, intensity: float, rng: np.random.Generator) -> int:
    """Return the index of an option drawn with the logit probabilities of its utility.

    Args:
        utilities:
            The utility of every option.

        intensity:
            The intensity of choice.

        rng:
            The random generator, such as `model.rng`.

    Returns:
        index:
            The chosen option.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.model import logit_choice
        >>> logit_choice([0.0, 100.0], 10.0, np.random.default_rng(0))
        1
    """
    probabilities = logit_probabilities(utilities, intensity)
    return int(rng.choice(len(probabilities), p=probabilities))


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


def _select(fitness: Array, count: int, select: str, rng: np.random.Generator, size: int) -> npt.NDArray[np.intp]:
    """Return the indices of the selected parents."""
    if select == 'tournament':
        entrants = rng.integers(len(fitness), size=(count, size))
        return entrants[np.arange(count), np.argmax(fitness[entrants], axis=1)]
    if select == 'roulette':
        weights = fitness - fitness.min()
        weights = np.ones_like(weights) if not weights.sum() else weights
        return rng.choice(len(fitness), size=count, p=weights / weights.sum())
    best = np.argsort(-fitness, kind='stable')[: max(1, len(fitness) // 2)]
    return best[rng.integers(len(best), size=count)]


def evolve(
    genomes: npt.ArrayLike,
    fitness: npt.ArrayLike,
    rng: np.random.Generator,
    *,
    select: str = 'tournament',
    crossover: float = 0.7,
    mutation: float = 0.01,
    elite: int = 0,
    tournament: int = 2,
    alphabet: int | None = 2,
    scale: float = 0.1,
) -> Array:
    """Return the next generation of strategies by selection, one-point crossover, and mutation.

    The operators draw from the model's random stream, so runs are reproducible. Discrete genomes over an alphabet,
    such as bit strings, mutate to a random other symbol, and real genomes, with `alphabet=None`, take Gaussian noise.

    Args:
        genomes:
            One row per agent, the strategy states of a group.

        fitness:
            The fitness of every agent.

        rng:
            The random generator, such as `model.rng`.

        select:
            The parent selection: `'tournament'`, `'roulette'`, or `'truncation'` to the better half.

        crossover:
            The probability that a pair of parents exchanges the tails after a random cut.

        mutation:
            The probability that each gene mutates.

        elite:
            The number of best genomes copied unchanged.

        tournament:
            The number of entrants of every tournament.

        alphabet:
            The number of symbols of a discrete genome, or `None` for a real genome.

        scale:
            The standard deviation of the mutation of a real genome.

    Returns:
        genomes:
            The genomes of the next generation, one row per agent.

    Raises:
        ValueError:
            The selection is unknown.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.model import evolve
        >>> genomes = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
        >>> evolve(genomes, [0.0, 1.0], np.random.default_rng(0), select='truncation', mutation=0.0).tolist()
        [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    """
    if select not in SELECTIONS:
        error_msg = f'Unknown selection {select!r}. Use one of {SELECTIONS}.'
        raise ValueError(error_msg)
    population = np.asarray(genomes, dtype=float)
    flat = population.reshape(len(population), -1)
    scores = np.asarray(fitness, dtype=float)
    n, length = flat.shape
    parents = flat[_select(scores, n, select, rng, tournament)].copy()
    for first in range(0, n - 1, 2):
        if length > 1 and rng.random() < crossover:
            cut = int(rng.integers(1, length))
            tail = parents[first, cut:].copy()
            parents[first, cut:] = parents[first + 1, cut:]
            parents[first + 1, cut:] = tail
    mutated = rng.random(parents.shape) < mutation
    if alphabet is None:
        parents = parents + mutated * rng.normal(0.0, scale, parents.shape)
    else:
        shift = rng.integers(1, max(alphabet, 2), parents.shape)
        parents = np.where(mutated, (parents + shift) % alphabet, parents)
    if elite:
        best = np.argsort(-scores, kind='stable')[:elite]
        parents[:elite] = flat[best]
    return parents.reshape(population.shape)
