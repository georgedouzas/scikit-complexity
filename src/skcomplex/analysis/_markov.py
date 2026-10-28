"""Analyse Markov chains: transition matrices of paths, stationary laws, spectral gaps, and Markov state models."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MarkovStateModel as DeeptimeModel
from deeptime.markov.msm import MaximumLikelihoodMSM
from scipy import linalg

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array
from ._base import BaseAnalysis, Source
from ._kinds import DataKind

TOLERANCE = 1e-10
KINDS = ('transition-matrix', 'stationary', 'spectral-gap', 'state-model', 'timescales')
MATRICES = ('transition', 'rate')
SETTINGS = {
    'transition-matrix': {'n_states', 'lag'},
    'stationary': {'matrix', 'n_states', 'lag'},
    'spectral-gap': {'matrix', 'n_states', 'lag', 'dt'},
    'state-model': {'lag', 'n_metastable', 'reversible', 'dt'},
    'timescales': {'lags', 'n_timescales', 'reversible', 'dt'},
}
"""Settings every kind reads, besides `kind` and `variables`."""


class MarkovAnalysis(BaseAnalysis):
    """Markov chain of discrete paths, or of a transition or rate matrix, of one kind.

    The kinds read the integer states of one path, or one row per path, or a matrix with `matrix`:

    - `'transition-matrix'`: the transitions after `lag` steps counted along the paths and normalized by row.
    - `'stationary'`: the stationary law `pi` of the matrix, `pi P = pi` or `pi Q = 0` for rates.
    - `'spectral-gap'`: the gap `1 - |lambda_2|`, or `-Re lambda_2` for rates, and the relaxation time of every mode.
    - `'state-model'`: the maximum-likelihood Markov state model of the largest connected set, estimated with
      deeptime, with its metastable sets found by PCCA+.
    - `'timescales'`: the slowest implied timescales of Markov state models at every lag of `lags`, which do not
      depend on the lag for a Markovian discretization.

    Args:
        kind:
            The quantity to compute, one of the kinds above.

        variables:
            The discrete state of a trajectory to read, or `None` for the first one.

        matrix:
            `None` to read paths, `'transition'` to read the source as a transition matrix, or `'rate'` to read it as
            the rate matrix of a continuous-time chain, whose diagonal is ignored.

        n_states:
            The number of states of the paths, or `None` for the largest state plus one.

        lag:
            The lag time in steps.

        lags:
            The lag times in steps of the implied timescales.

        n_timescales:
            The number of implied timescales.

        n_metastable:
            The number of metastable sets PCCA+ groups the states into, or `None` for none.

        reversible:
            Whether the state models obey detailed balance, which PCCA+ needs.

        dt:
            The time of one step.

    Attributes:
        path_:
            The integer states of the paths, one row per path, read without a matrix.

        matrix_:
            The matrix read with `matrix`.

        counts_:
            The transition counts, one row per origin, of `'transition-matrix'`.

        transition_matrix_:
            The probability of every destination after one lag from every origin, of `'transition-matrix'` and
            `'state-model'`.

        stationary_:
            The stationary law, of `'stationary'` and `'state-model'`.

        spectral_gap_:
            The spectral gap, of `'spectral-gap'`.

        relaxation_times_:
            `-dt / ln |lambda|`, or `-1 / Re lambda` for rates, of every mode slowest first, of `'spectral-gap'`.

        states_:
            The states of the largest connected set, which index the rows of the matrix, of `'state-model'`.

        timescales_:
            The implied timescales `-lag dt / ln |lambda_i|` slowest first, of `'state-model'`, and one row per lag
            of `'timescales'`.

        metastable_sets_:
            The states of every metastable set, empty without a number of sets, of `'state-model'`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import MarkovAnalysis
        >>> MarkovAnalysis().check().prepare([0, 1, 0, 1, 1]).run().transition_matrix_.tolist()
        [[0.0, 1.0], [0.5, 0.5]]
        >>> stationary = MarkovAnalysis('stationary', matrix='transition')
        >>> stationary.check().prepare([[0.5, 0.5], [0.25, 0.75]]).run().stationary_.round(3).tolist()
        [0.333, 0.667]
        >>> path = np.random.default_rng(0).integers(0, 2, 10_000)
        >>> MarkovAnalysis('state-model').check().prepare(path).run().transition_matrix_.round(1).tolist()
        [[0.5, 0.5], [0.5, 0.5]]
    """

    def __init__(
        self,
        kind: str = 'transition-matrix',
        variables: Sequence[str] | None = None,
        *,
        matrix: str | None = None,
        n_states: int | None = None,
        lag: int = 1,
        lags: Sequence[int] = (1,),
        n_timescales: int = 1,
        n_metastable: int | None = None,
        reversible: bool = True,
        dt: float = 1.0,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.matrix = matrix
        self.n_states = n_states
        self.lag = lag
        self.lags = lags
        self.n_timescales = n_timescales
        self.n_metastable = n_metastable
        self.reversible = reversible
        self.dt = dt

    def _check(self) -> Self:
        """Validate the kind, the matrix, the lags, and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        if self.matrix is not None:
            self._check_choice('matrix', self.matrix, MATRICES)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        if self.lag < 1 or any(lag < 1 for lag in self.lags) or self.n_timescales < 1:
            error_msg = (
                f'{type(self).__name__} has lag={self.lag!r}, lags={self.lags!r}, and '
                f'n_timescales={self.n_timescales!r}. Use 1 or more.'
            )
            raise AnalysisError(error_msg)
        if self.dt <= 0:
            error_msg = f'{type(self).__name__} has dt={self.dt!r}. Use a positive time step.'
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the path kind."""
        return DataKind.PATH

    def _prepare(self, source: Source) -> Self:
        """Read the matrix with `matrix`, or the states of the paths, a trajectory through its jump path."""
        if self.matrix is not None:
            self.matrix_ = np.asarray(source, dtype=float)
            return self
        if isinstance(source, Trajectory):
            name = self.variables[0] if self.variables else None
            self.path_ = self._paths(self._read_path(source, name).states)
            return self
        self.path_ = self._paths(source)
        return self

    def _get_matrix(self) -> Array:
        """Return the matrix read, or the transition matrix estimated from the paths."""
        if self.matrix is not None:
            return self.matrix_
        return self._transition_matrix(self.path_, self.n_states, self.lag)

    def _run_state_model(self) -> None:
        """Estimate the Markov state model and its metastable sets."""
        model = self._fit(self._get_states(), self.lag, reversible=self.reversible)
        symbols = np.asarray(model.count_model.state_symbols, dtype=np.int64)
        self.states_ = symbols
        self.transition_matrix_ = np.asarray(model.transition_matrix, dtype=float)
        self.stationary_ = np.asarray(model.stationary_distribution, dtype=float)
        self.timescales_ = np.asarray(model.timescales(), dtype=float) * self.dt
        self.metastable_sets_ = []
        if self.n_metastable is not None:
            self.metastable_sets_ = [
                symbols[np.asarray(found, dtype=np.int64)] for found in model.pcca(self.n_metastable).sets
            ]

    def _get_states(self) -> npt.NDArray[np.int64]:
        """Return one path as a row, or several paths as rows, as deeptime reads them."""
        return self.path_[0] if self.path_.shape[0] == 1 else self.path_

    def _run(self, inputs: object = None) -> Self:
        """Compute the quantity of the kind."""
        rates = self.matrix == 'rate'
        if self.kind == 'transition-matrix':
            self.counts_ = self._transition_counts(self.path_, self.n_states, self.lag)
            self.transition_matrix_ = self._transition_matrix(self.path_, self.n_states, self.lag)
        elif self.kind == 'stationary':
            self.stationary_ = self._stationary_distribution(self._get_matrix(), rates=rates)
        elif self.kind == 'spectral-gap':
            matrix = self._get_matrix()
            self.spectral_gap_ = self._spectral_gap(matrix, rates=rates)
            self.relaxation_times_ = self._relaxation_times(matrix, rates=rates, dt=self.dt)
        elif self.kind == 'state-model':
            self._run_state_model()
        else:
            states = self._get_states()
            self.timescales_ = np.asarray(
                [
                    self._fit(states, lag, reversible=self.reversible).timescales(self.n_timescales) * self.dt
                    for lag in self.lags
                ],
                dtype=float,
            )
        return self

    @staticmethod
    def _paths(states: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Return integer states as one row per path."""
        array = np.asarray(states, dtype=np.int64)
        return array.reshape(1, -1) if array.ndim == 1 else array

    @staticmethod
    def _generator(matrix: npt.ArrayLike, *, rates: bool) -> Array:
        """Return a rate matrix with diagonal minus the escape rates, or a transition matrix minus the identity."""
        array = np.asarray(matrix, dtype=float)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:  # noqa: PLR2004
            error_msg = f'A transition or rate matrix is square, not of shape {array.shape}.'
            raise AnalysisError(error_msg)
        if not rates:
            return array - np.eye(array.shape[0])
        off = array - np.diag(np.diag(array))
        return off - np.diag(off.sum(axis=1))

    @staticmethod
    def _eigenvalues(matrix: npt.ArrayLike, *, rates: bool) -> npt.NDArray[np.complex128]:
        """Return the eigenvalues of the generator, the stationary 0 first, then by falling real part."""
        values = linalg.eigvals(MarkovAnalysis._generator(matrix, rates=rates))
        return values[np.argsort(-values.real)]

    @staticmethod
    def _transition_counts(states: npt.ArrayLike, n_states: int | None = None, lag: int = 1) -> npt.NDArray[np.int64]:
        """Return the number of transitions from every state to every state after `lag` steps along discrete paths."""
        paths = MarkovAnalysis._paths(states)
        size = int(paths.max()) + 1 if n_states is None else n_states
        origins, destinations = paths[:, :-lag].ravel(), paths[:, lag:].ravel()
        return np.bincount(origins * size + destinations, minlength=size * size).reshape(size, size)

    @staticmethod
    def _transition_matrix(states: npt.ArrayLike, n_states: int | None = None, lag: int = 1) -> Array:
        """Return the counts normalized by row, the identity row for unvisited states."""
        counts = MarkovAnalysis._transition_counts(states, n_states, lag).astype(float)
        totals = counts.sum(axis=1)
        unvisited = totals == 0
        counts[unvisited, unvisited] = 1.0
        return counts / counts.sum(axis=1, keepdims=True)

    @staticmethod
    def _stationary_distribution(matrix: npt.ArrayLike, *, rates: bool = False) -> Array:
        """Return the stationary law `pi` of a transition matrix, `pi P = pi`, or of a rate matrix, `pi Q = 0`."""
        generator = MarkovAnalysis._generator(matrix, rates=rates)
        null = linalg.null_space(generator.T, rcond=TOLERANCE)
        vector = np.abs(null[:, 0]) if null.size else np.abs(linalg.eig(generator.T)[1][:, 0].real)
        return vector / vector.sum()

    @staticmethod
    def _spectral_gap(matrix: npt.ArrayLike, *, rates: bool = False) -> float:
        """Return `1 - |lambda_2|` of a transition matrix or `-Re lambda_2` of a rate matrix."""
        values = MarkovAnalysis._eigenvalues(matrix, rates=rates)
        if rates:
            return float(-values[1].real)
        return float(1 - np.abs(values[1:] + 1).max())

    @staticmethod
    def _relaxation_times(matrix: npt.ArrayLike, *, rates: bool = False, dt: float = 1.0) -> Array:
        """Return the relaxation times of every mode but the stationary one, slowest first."""
        values = MarkovAnalysis._eigenvalues(matrix, rates=rates)[1:]
        if rates:
            return -1 / values.real
        moduli = np.clip(np.abs(values + 1), 1e-300, 1 - 1e-15)
        return np.sort(-dt / np.log(moduli))[::-1]

    @staticmethod
    def _fit(states: npt.ArrayLike, lag: int, *, reversible: bool) -> DeeptimeModel:
        """Return the maximum-likelihood model of the largest connected set of states at a lag."""
        paths = np.asarray(states, dtype=np.int32)
        trajectories = list(paths) if paths.ndim > 1 else paths
        counts = TransitionCountEstimator(lagtime=lag, count_mode='sliding').fit(trajectories).fetch_model()
        return MaximumLikelihoodMSM(reversible=reversible).fit(counts.submodel_largest()).fetch_model()
