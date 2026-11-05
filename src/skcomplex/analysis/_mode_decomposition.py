"""Fit linear surrogates to series by dynamic mode decomposition, on the variables or on a dictionary of observables."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace

import numpy as np
from sklearn.base import TransformerMixin, clone

from ..core.laws import BaseLaw, RateLaw
from ..core.spaces import ContinuousTime
from ..engine._utils import vector
from ..errors import AnalysisError
from ..specs import SamplesSpec
from ..types import Array, ComplexArray, DataKind
from ._base import BaseAnalysis
from ._sparse_identification import SparseIdentificationAnalysis
from .types import Runs

SMALL_EXPONENT = 1e-12


class ModeDecompositionAnalysis(BaseAnalysis):
    """Linear system fitted to series by dynamic mode decomposition, computed with PyDMD.

    The decomposition fits the exact DMD of the one-sample map `g(t + dt) = A g(t)` to the pairs of consecutive samples
    of every run. With a dictionary, `g` holds the observables the dictionary computes from the variables, which is
    extended DMD, and the dictionary should keep the variables among its observables. The fitted system follows the
    rate law `dg/dt = L g` with `L = Phi diag(ln(mu) / dt) Phi^+` under continuous time and starts at the first sample
    of the first run. The forecast iterates the map from the last sample of the last run.

    With inputs, the decomposition is dynamic mode decomposition with control, `g(t + dt) = A g(t) + B u(t)`, where `g`
    holds the variables that are not inputs or their observables. The fitted system follows `dg/dt = L g + M u`, where
    `M` gives the same one-sample map under an input held over the sample. Every input is a bounded adaptive parameter
    of the surrogate element, bounded by its range in the samples, and the forecast holds the inputs at their last
    values.

    Args:
        variables:
            The variables of trajectories, or `None` for every scalar state or observable. The inputs join them when
            they are not listed.

        rank:
            The rank of the truncated singular value decomposition: a number of modes, 0 for the optimal hard
            threshold, a share of the energy below 1, or `None` to keep every singular value.

        dictionary:
            A scikit-learn transformer that computes the observables, such as `PolynomialFeatures(2)`, or `None` for
            the variables themselves.

        inputs:
            The names of the variables that are control inputs, or `None` for none.

        validation:
            The share of the samples of every run held out at its end to measure the validation error.

        horizon:
            The number of samples to forecast after the last sample of the last run.

        dt:
            The time between the rows of an array.

        label:
            The label of the surrogate element.

    Attributes:
        samples_:
            The series of every run, the names of the variables, and the step, set by `prepare`.

        system_:
            The system of one surrogate element under continuous time, whose states follow `dg/dt = L g + M u`.

        labels_:
            The variables, or the observables of the dictionary, in the order of the states.

        states_:
            The names of the states of the surrogate element.

        last_state_:
            The values of the states at the last sample of the last run, where the forecast starts.

        dt_:
            The time between the samples it was fitted to.

        training_error_:
            The relative error of the predicted increments over one sample on the training samples.

        validation_error_:
            The relative error on the held-out samples, NaN when none are held out.

        eigenvalues_:
            The continuous eigenvalues `ln(mu) / dt` of the modes, from the eigenvalues `mu` of the one-sample map.

        modes_:
            The modes, one column per mode and one row per state.

        amplitudes_:
            The amplitudes of the modes at the last sample of the last run.

        generator_:
            The matrix `L` of the rate law, one row and one column per state.

        input_generator_:
            The matrix `M` of the rate law, one row per state and one column per input.

        operator_:
            The matrix `A` of the one-sample map, one row and one column per state.

        input_matrix_:
            The matrix `B` of the one-sample map, one row per state and one column per input.

        dictionary_:
            The fitted dictionary, which lifts new states to the observables, or `None` without a dictionary.

        inputs_:
            The names of the adaptive parameters of the surrogate element that hold the inputs.

        forecast_:
            The states at the `horizon` samples after the last one, one row per sample.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import ModeDecompositionAnalysis
        >>> times = np.arange(0.0, 10.0, 0.1)
        >>> values = np.column_stack([np.cos(times), np.sin(times)])
        >>> analysis = ModeDecompositionAnalysis(['x', 'y'], horizon=5, dt=0.1).prepare(values).run()
        >>> np.sort(analysis.eigenvalues_.imag).round(6).tolist()
        [-1.0, 1.0]
        >>> bool(np.allclose(analysis.forecast_[:, 0], np.cos(times[-1] + 0.1 * np.arange(1, 6))))
        True
        >>> u = np.random.default_rng(0).uniform(-1.0, 1.0, 100)
        >>> x = np.zeros(100)
        >>> for k in range(99):
        ...     x[k + 1] = 0.9 * x[k] + 0.5 * u[k]
        >>> analysis = ModeDecompositionAnalysis(['x', 'u'], inputs=['u'], validation=0.0)
        >>> analysis = analysis.prepare(np.column_stack([x, u])).run()
        >>> analysis.operator_.round(6).tolist(), analysis.input_matrix_.round(6).tolist()
        ([[0.9]], [[0.5]])
    """

    def __init__(
        self,
        variables: Sequence[str] | None = None,
        *,
        rank: float | None = None,
        dictionary: TransformerMixin | None = None,
        inputs: Sequence[str] | None = None,
        validation: float = 0.2,
        horizon: int = 0,
        dt: float = 1.0,
        label: str = 'surrogate',
    ) -> None:
        self.variables = variables
        self.rank = rank
        self.dictionary = dictionary
        self.inputs = inputs
        self.validation = validation
        self.horizon = horizon
        self.dt = dt
        self.label = label

    def _check(self) -> None:
        """Raise an error for a validation share outside `[0, 1)` or a negative horizon."""
        SparseIdentificationAnalysis._check_validation(self.validation)
        SparseIdentificationAnalysis._check_horizon(self.horizon)

    def _get_data_kind(self) -> DataKind:
        """Return the series kind."""
        return DataKind.SERIES

    def _prepare(self, source: Runs) -> None:
        """Read the uniformly sampled series of every run, with the inputs among the variables."""
        added = [name for name in self.inputs or () if name not in (self.variables or ())]
        variables = None if self.variables is None else [*self.variables, *added]
        self.samples_ = self._read_samples(source, variables, self.dt)

    def _split_inputs(self, samples: SamplesSpec) -> tuple[SamplesSpec, list[str], list[Array]]:
        """Return the samples without the inputs, the labels of the inputs, and their series.

        Args:
            samples:
                The series of every run with the inputs among the variables.

        Returns:
            parts:
                The samples of the other variables, the labels of the inputs, and one array per run, with one column per
                input.

        Raises:
            AnalysisError:
                An input is not a variable of the series.
        """
        inputs = list(self.inputs or ())
        missing = sorted(set(inputs) - set(samples.labels))
        if missing:
            error_msg = f'The inputs {missing} are not variables of the series, which are {samples.labels}.'
            raise AnalysisError(error_msg)
        others = [index for index, label in enumerate(samples.labels) if label not in inputs]
        columns = [samples.labels.index(name) for name in inputs]
        labels = [samples.labels[index] for index in others]
        return (
            SamplesSpec([values[:, others] for values in samples.series], labels, samples.dt),
            inputs,
            [values[:, columns] for values in samples.series],
        )

    def _fit_operators(
        self,
        before: Array,
        after: Array,
        controls: Array,
    ) -> tuple[Array, Array, ComplexArray, ComplexArray]:
        """Return the matrices `A` and `B` of the one-sample map, the modes, and the eigenvalues, fitted with PyDMD.

        Args:
            before:
                The states before every pair of samples, one column per pair.

            after:
                The states after every pair of samples, one column per pair.

            controls:
                The inputs of every pair of samples, one column per pair and one row per input.

        Returns:
            operators:
                The matrices `A` and `B`, the modes, and the eigenvalues of `A`.
        """
        rank = -1 if self.rank is None else self.rank
        pydmd = self._import('pydmd')
        if not len(controls):
            fitted = pydmd.DMD(svd_rank=rank, exact=True).fit(before, after)
            modes = np.asarray(fitted.modes, dtype=complex)
            eigenvalues = np.asarray(fitted.eigs, dtype=complex)
            operator = np.real(modes @ np.diag(eigenvalues) @ np.linalg.pinv(modes))
            return operator, np.zeros((len(before), 0)), modes, eigenvalues
        rank_with_inputs = rank + len(controls) if rank >= 1 else rank
        fitted = pydmd.DMDc(svd_rank=rank, svd_rank_omega=rank_with_inputs, lag=before.shape[1])
        fitted.fit(np.hstack([before, after]), controls)
        basis = np.asarray(fitted.basis)
        operator = np.real(basis @ np.asarray(fitted.operator.as_numpy_array) @ basis.conj().T)
        modes = np.asarray(fitted.modes, dtype=complex)
        return operator, np.real(np.asarray(fitted.B)), modes, np.asarray(fitted.eigs, dtype=complex)

    def _run(self, inputs: object = None) -> None:
        """Fit the decomposition and set the surrogate system, its errors, its modes, and the forecast."""
        samples, input_labels, controls = self._split_inputs(self.samples_)
        self.dictionary_ = (
            None if self.dictionary is None else clone(self.dictionary).fit(np.concatenate(samples.series))
        )
        lifted, labels, states = self._lift(samples, self.dictionary_)
        training, held = SparseIdentificationAnalysis._split_samples(lifted, self.validation)
        driven, driven_held = SparseIdentificationAnalysis._split_samples(controls, self.validation)
        before = np.concatenate([values[:-1] for values in training]).T
        after = np.concatenate([values[1:] for values in training]).T
        applied = np.concatenate([values[:-1] for values in driven]).T
        operator, input_matrix, modes, multipliers = self._fit_operators(before, after, applied)
        eigenvalues: ComplexArray = np.asarray(np.log(multipliers) / samples.dt, dtype=complex)
        generator = np.real(modes @ np.diag(eigenvalues) @ np.linalg.pinv(modes))
        input_generator = self._input_generator(modes, multipliers, eigenvalues, input_matrix, samples.dt)
        names = self._input_names(input_labels, states)
        self.system_ = SparseIdentificationAnalysis._surrogate_system(
            self.label,
            dict(zip(states, map(float, lifted[0][0]), strict=True)),
            self._linear_laws(states, generator, names, input_generator),
            ContinuousTime(),
            inputs=self._input_declarations(names, controls),
        )
        self.labels_ = labels
        self.states_ = states
        self.inputs_ = names
        self.last_state_ = np.asarray(lifted[-1][-1], dtype=float)
        self.dt_ = samples.dt
        self.training_error_ = SparseIdentificationAnalysis._relative_error(
            *self._increments(operator, input_matrix, training, driven),
        )
        self.validation_error_ = SparseIdentificationAnalysis._relative_error(
            *self._increments(operator, input_matrix, held, driven_held),
        )
        self.eigenvalues_ = eigenvalues
        self.modes_ = modes
        self.amplitudes_ = np.linalg.lstsq(modes, self.last_state_.astype(complex), rcond=None)[0]
        self.generator_ = generator
        self.input_generator_ = input_generator
        self.operator_ = operator
        self.input_matrix_ = input_matrix
        self.forecast_ = self._forecast(operator, input_matrix, controls[-1][-1])

    def _forecast(self, operator: Array, input_matrix: Array, last_input: Array) -> Array:
        """Return the states at the samples of the horizon, with the inputs held at their last values."""
        current = self.last_state_
        rows = []
        for _ in range(self.horizon):
            current = operator @ current + input_matrix @ last_input
            rows.append(current)
        return np.asarray(rows, dtype=float).reshape(self.horizon, len(current))

    @staticmethod
    def _input_generator(
        modes: ComplexArray,
        multipliers: ComplexArray,
        eigenvalues: ComplexArray,
        input_matrix: Array,
        dt: float,
    ) -> Array:
        """Return the matrix `M` of the rate law whose flow over a sample under a held input gives the matrix `B`."""
        slow = np.abs(eigenvalues) * dt < SMALL_EXPONENT
        safe = np.where(slow, 1.0, eigenvalues)
        weights = np.where(slow, dt, (multipliers - 1.0) / safe)
        integral = np.real(modes @ np.diag(weights) @ np.linalg.pinv(modes))
        return np.linalg.lstsq(integral, input_matrix, rcond=None)[0]

    @staticmethod
    def _input_names(labels: Sequence[str], states: Sequence[str]) -> list[str]:
        """Return the parameter names of the inputs, which differ from the names of the states."""
        names = SparseIdentificationAnalysis._state_names(labels)
        return [f'{name}_input' if name in states else name for name in names]

    @staticmethod
    def _input_declarations(
        names: Sequence[str],
        controls: Sequence[Array],
    ) -> dict[str, tuple[float, tuple[float, float]]]:
        """Return the first value and the range in the samples of every input, by name."""
        joined = np.concatenate(controls)
        return {
            name: (float(controls[0][0, index]), (float(joined[:, index].min()), float(joined[:, index].max())))
            for index, name in enumerate(names)
        }

    @staticmethod
    def _lift(samples: SamplesSpec, fitted: TransformerMixin | None) -> tuple[list[Array], list[str], list[str]]:
        """Return the series, labels, and state names of the observables: the variables without a dictionary."""
        names = SparseIdentificationAnalysis._state_names(samples.labels)
        if fitted is None:
            return samples.series, samples.labels, names
        lifted = [np.asarray(fitted.transform(values), dtype=float) for values in samples.series]
        labels = [str(name) for name in fitted.get_feature_names_out(samples.labels)]
        return (
            lifted,
            labels,
            SparseIdentificationAnalysis._state_names([str(name) for name in fitted.get_feature_names_out(names)]),
        )

    @staticmethod
    def _increments(
        operator: Array,
        input_matrix: Array,
        parts: list[Array],
        controls: list[Array],
    ) -> tuple[list[Array], list[Array]]:
        """Return the increments over one sample that the one-sample map predicts and those of the series."""
        predicted = [
            values[:-1] @ (operator - np.eye(len(operator))).T + driven[:-1] @ input_matrix.T
            for values, driven in zip(parts, controls, strict=True)
        ]
        return predicted, [np.diff(values, axis=0) for values in parts]

    @staticmethod
    def _linear_laws(
        names: Sequence[str],
        generator: Array,
        inputs: Sequence[str],
        input_generator: Array,
    ) -> Callable[[SimpleNamespace], list[BaseLaw]]:
        """Return the function that gives the rate law `dg/dt = L g + M u` of the states named."""
        matrix = np.hstack([generator, input_generator])

        def laws(s: SimpleNamespace) -> list[BaseLaw]:
            """Return the matrix rate law."""
            symbols = SparseIdentificationAnalysis._symbols_of(s, names)
            read = [*symbols, *SparseIdentificationAnalysis._symbols_of(s, inputs)]
            rows = [sum(float(value) * item for value, item in zip(row, read, strict=True) if value) for row in matrix]
            return [RateLaw(vector(symbols), vector(rows))]

        return laws
