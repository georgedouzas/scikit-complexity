"""Estimate Markov state models of discrete paths with implied timescales, metastable sets, and committors."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import deeptime.markov.tools.analysis as markov_analysis
import numpy as np
import numpy.typing as npt
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MarkovStateModel as DeeptimeModel
from deeptime.markov.msm import MaximumLikelihoodMSM

from ._kinds import DataKind, analysis

type Array = npt.NDArray[np.float64]


def _fit(states: npt.ArrayLike, lag: int, *, reversible: bool) -> DeeptimeModel:
    """Return the maximum-likelihood model of the largest connected set of states at a lag."""
    paths = np.asarray(states, dtype=np.int32)
    trajectories = list(paths) if paths.ndim > 1 else paths
    counts = TransitionCountEstimator(lagtime=lag, count_mode='sliding').fit(trajectories).fetch_model()
    return MaximumLikelihoodMSM(reversible=reversible).fit(counts.submodel_largest()).fetch_model()


@dataclass(frozen=True)
class MarkovStateModel:
    """Markov state model of discrete paths at a lag time.

    Args:
        states:
            The states of the largest connected set, which index the rows of the matrix.

        transition_matrix:
            The probability of every destination after one lag from every origin.

        stationary:
            The stationary distribution over the states.

        timescales:
            The implied timescales `-lag dt / ln |lambda_i|` of the slow modes, slowest first.

        metastable_sets:
            The states of every metastable set found by PCCA+, empty without a number of sets.

        lag:
            The lag time in steps.
    """

    states: npt.NDArray[np.int64]
    transition_matrix: Array
    stationary: Array
    timescales: Array
    metastable_sets: list[npt.NDArray[np.int64]]
    lag: int


@analysis(DataKind.PATH)
def markov_state_model(
    states: npt.ArrayLike,
    lag: int = 1,
    *,
    n_metastable: int | None = None,
    reversible: bool = True,
    dt: float = 1.0,
) -> MarkovStateModel:
    """Return the maximum-likelihood Markov state model of discrete paths, estimated with deeptime.

    Args:
        states:
            The integer states of one path, or one row per path, such as bins of a coordinate.

        lag:
            The lag time in steps.

        n_metastable:
            The number of metastable sets PCCA+ groups the states into, or `None` for none.

        reversible:
            Whether the model obeys detailed balance, which PCCA+ needs.

        dt:
            The time of one step.

    Returns:
        model:
            The states, the transition matrix, the stationary law, the timescales, and the metastable sets.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import markov_state_model
        >>> path = np.random.default_rng(0).integers(0, 2, 10_000)
        >>> markov_state_model(path).transition_matrix.round(1).tolist()
        [[0.5, 0.5], [0.5, 0.5]]
    """
    model = _fit(states, lag, reversible=reversible)
    sets: list[npt.NDArray[np.int64]] = []
    if n_metastable is not None:
        symbols = np.asarray(model.count_model.state_symbols, dtype=np.int64)
        sets = [symbols[np.asarray(found, dtype=np.int64)] for found in model.pcca(n_metastable).sets]
    return MarkovStateModel(
        np.asarray(model.count_model.state_symbols, dtype=np.int64),
        np.asarray(model.transition_matrix, dtype=float),
        np.asarray(model.stationary_distribution, dtype=float),
        np.asarray(model.timescales(), dtype=float) * dt,
        sets,
        lag,
    )


@analysis(DataKind.PATH)
def implied_timescales(
    states: npt.ArrayLike,
    lags: Sequence[int],
    k: int = 1,
    *,
    reversible: bool = True,
    dt: float = 1.0,
) -> Array:
    """Return the `k` slowest implied timescales of Markov state models estimated at every lag.

    A Markovian discretization gives timescales that do not depend on the lag.

    Args:
        states:
            The integer states of one path, or one row per path.

        lags:
            The lag times in steps.

        k:
            The number of timescales.

        reversible:
            Whether the models obey detailed balance.

        dt:
            The time of one step.

    Returns:
        timescales:
            One row per lag and one column per timescale, slowest first.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import implied_timescales
        >>> path = (np.arange(10_000) // 50) % 2
        >>> implied_timescales(path, [1, 2]).shape
        (2, 1)
    """
    return np.asarray([_fit(states, lag, reversible=reversible).timescales(k) * dt for lag in lags], dtype=float)


@analysis(DataKind.PATH)
def committor(matrix: npt.ArrayLike, source: Sequence[int], target: Sequence[int]) -> Array:
    """Return the forward committor, the probability that a chain from every state reaches the target before the source.

    Args:
        matrix:
            The transition matrix.

        source:
            The states of the source set.

        target:
            The states of the target set.

    Returns:
        committor:
            The probability for every state, 0 on the source and 1 on the target.

    Examples:
        >>> from skcomplex.core.analysis import committor
        >>> committor([[0.5, 0.5, 0.0], [0.25, 0.5, 0.25], [0.0, 0.5, 0.5]], [0], [2]).tolist()
        [0.0, 0.5, 1.0]
    """
    return np.asarray(markov_analysis.committor(np.asarray(matrix, dtype=float), list(source), list(target)))


@analysis(DataKind.PATH)
def mean_first_passage_time(
    matrix: npt.ArrayLike,
    source: Sequence[int],
    target: Sequence[int],
    *,
    lag: int = 1,
    dt: float = 1.0,
) -> float:
    """Return the mean first-passage time from a source set, started in its stationary weights, to a target set.

    Args:
        matrix:
            The transition matrix at the lag.

        source:
            The states of the source set.

        target:
            The states of the target set.

        lag:
            The lag of the matrix in steps.

        dt:
            The time of one step.

    Returns:
        time:
            The mean first-passage time.

    Examples:
        >>> from skcomplex.core.analysis import mean_first_passage_time
        >>> mean_first_passage_time([[0.5, 0.5], [0.5, 0.5]], [0], [1])
        2.0
    """
    array = np.asarray(matrix, dtype=float)
    return float(markov_analysis.mfpt(array, list(target), origin=list(source), tau=lag) * dt)
