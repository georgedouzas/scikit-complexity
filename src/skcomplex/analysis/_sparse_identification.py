"""Fit sparse symbolic surrogates to series by sparse identification of nonlinear dynamics."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
import keyword
import math
import re
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ..core.elements import Element
from ..core.laws import BaseLaw, RateLaw
from ..core.system import System
from ..core.times import BaseTime, ContinuousTime
from ..engine.runtime import Simulation
from ..errors import AnalysisError
from ..specs import StateSpec
from ..types import Array, DataKind
from ._base import BaseAnalysis
from .types import Runs

DIFFERENCE_POINTS = 3
RESERVED_NAMES = frozenset({'label', 'laws', 'visual', 'group', 'define'})


class SparseIdentificationAnalysis(BaseAnalysis):
    """Sparse symbolic system fitted to series by sparse identification of nonlinear dynamics, computed with PySINDy.

    SINDy writes the derivative of every variable as a sparse sum of the monomials of the variables up to a degree. It
    estimates the derivatives by centred finite differences and selects the terms by sequentially thresholded least
    squares. The fitted system holds one rate law per variable under continuous time and starts at the first sample of
    the first run, and the forecast simulates it from the last sample of the last run.

    Args:
        variables:
            The variables of trajectories, or `None` for every scalar state or observable.

        degree:
            The highest degree of the candidate monomials.

        threshold:
            The smallest magnitude of a kept coefficient.

        alpha:
            The strength of the ridge penalty of every least-squares fit.

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
            The system of one surrogate element under continuous time, with one rate law per variable.

        labels_:
            The variables the system was fitted to, in the order of its states.

        states_:
            The names of the states of the surrogate element.

        last_state_:
            The values of the states at the last sample of the last run, where the forecast starts.

        dt_:
            The time between the samples it was fitted to.

        training_error_:
            The relative error of the predicted derivatives on the training samples.

        validation_error_:
            The relative error on the held-out samples, NaN when none are held out.

        features_:
            The candidate monomials of the states, such as `x y`.

        coefficients_:
            The coefficient of every monomial in the rate of every state, one row per state and one column per
            monomial, zero where the term is absent.

        forecast_:
            The variables at the `horizon` samples after the last one, one row per sample.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import SparseIdentificationAnalysis
        >>> times = np.arange(0.0, 2.0, 0.001)
        >>> analysis = SparseIdentificationAnalysis(['x'], degree=1, dt=0.001)
        >>> analysis = analysis.check().prepare(np.exp(-2 * times)).run()
        >>> analysis.features_, analysis.coefficients_.round(3).tolist()
        (['1', 'x'], [[0.0, -2.0]])
    """

    def __init__(
        self,
        variables: Sequence[str] | None = None,
        *,
        degree: int = 2,
        threshold: float = 0.1,
        alpha: float = 0.05,
        validation: float = 0.2,
        horizon: int = 0,
        dt: float = 1.0,
        label: str = 'surrogate',
    ) -> None:
        self.variables = variables
        self.degree = degree
        self.threshold = threshold
        self.alpha = alpha
        self.validation = validation
        self.horizon = horizon
        self.dt = dt
        self.label = label

    def _check(self) -> None:
        """Raise an error for a validation share outside `[0, 1)` or a negative horizon."""
        self._check_validation(self.validation)
        self._check_horizon(self.horizon)

    def _get_data_kind(self) -> DataKind:
        """Return the series kind."""
        return DataKind.SERIES

    def _prepare(self, source: Runs) -> None:
        """Read the uniformly sampled series of every run."""
        self.samples_ = self._read_samples(source, self.variables, self.dt)

    def _run(self, inputs: object = None) -> None:
        """Fit the sparse model and set the surrogate system, its errors, its coefficients, and the forecast."""
        samples = self.samples_
        states = self._state_names(samples.labels)
        training, held = self._split_samples(samples.series, self.validation)
        sindy = self._import('pysindy')
        model = sindy.SINDy(
            optimizer=sindy.STLSQ(threshold=self.threshold, alpha=self.alpha),
            feature_library=sindy.PolynomialLibrary(degree=self.degree),
        ).fit(training, t=samples.dt, feature_names=states)
        coefficients = np.asarray(model.coefficients(), dtype=float)
        powers = np.asarray(model.feature_library.powers_, dtype=int)
        self.system_ = self._surrogate_system(
            'SparseSurrogate',
            self.label,
            dict(zip(states, map(float, samples.series[0][0]), strict=True)),
            self._sparse_laws(states, coefficients, powers),
            ContinuousTime(),
        )
        self.labels_ = samples.labels
        self.states_ = states
        self.last_state_ = np.asarray(samples.series[-1][-1], dtype=float)
        self.dt_ = samples.dt
        predict, differentiate = model.predict, model.differentiation_method
        self.training_error_ = self._relative_error(*self._derivatives(predict, differentiate, training, samples.dt))
        self.validation_error_ = self._relative_error(*self._derivatives(predict, differentiate, held, samples.dt))
        self.features_ = [str(name) for name in model.get_feature_names()]
        self.coefficients_ = coefficients
        self.forecast_ = self._forecast()

    def _forecast(self) -> Array:
        """Return the variables of the fitted system simulated from the last sample, one row per sample."""
        names = [f'{self.label}.{name}' for name in self.states_]
        simulation = (
            Simulation(
                self.system_,
                dt=self.dt_,
                initial_state=dict(zip(names, map(float, self.last_state_), strict=True)),
            )
            .check()
            .prepare()
        )
        trajectory = simulation.run(until=self.horizon * self.dt_).trajectory_
        values = np.column_stack([np.asarray(trajectory.values(name), dtype=np.float64) for name in names])
        return values[1 : self.horizon + 1].reshape(self.horizon, len(names))

    @staticmethod
    def _term(symbols: Sequence[Expression], powers: npt.NDArray[np.int_]) -> Expression:
        """Return the monomial of the symbols with the given powers."""
        return np.prod([item ** int(power) for item, power in zip(symbols, powers, strict=True) if power], initial=1)

    @staticmethod
    def _sparse_laws(
        names: Sequence[str],
        coefficients: Array,
        powers: npt.NDArray[np.int_],
    ) -> Callable[[SimpleNamespace], list[BaseLaw]]:
        """Return the function that gives one rate law per state, a sum of the monomials with nonzero coefficients."""

        def laws(s: SimpleNamespace) -> list[BaseLaw]:
            """Return the sparse rate laws."""
            symbols = SparseIdentificationAnalysis._symbols_of(s, names)
            terms = [SparseIdentificationAnalysis._term(symbols, row) for row in powers]
            rates = [
                sum(float(value) * term for value, term in zip(row, terms, strict=True) if value)
                for row in coefficients
            ]
            return [RateLaw(item, rate) for item, rate in zip(symbols, rates, strict=True)]

        return laws

    @staticmethod
    def _derivatives(
        predict: Callable[[Array], npt.ArrayLike],
        differentiate: Callable[[Array, float], npt.ArrayLike],
        parts: list[Array],
        dt: float,
    ) -> tuple[list[Array], list[Array]]:
        """Return the derivatives a model predicts and those its differentiation method estimates from the series."""
        usable = [values for values in parts if len(values) >= DIFFERENCE_POINTS]
        predicted = [np.asarray(predict(values), dtype=np.float64) for values in usable]
        measured = [np.asarray(differentiate(values, dt), dtype=np.float64) for values in usable]
        return predicted, measured

    @staticmethod
    def _identifier(text: str) -> str:
        """Return a valid state name made from a label, which is neither a keyword nor a name an element reserves."""
        cleaned = re.sub(r'\W+', '_', text).strip('_') or 'v'
        cleaned = f'v_{cleaned}' if cleaned[0].isdigit() else cleaned
        return f'{cleaned}_' if keyword.iskeyword(cleaned) or cleaned in RESERVED_NAMES else cleaned

    @staticmethod
    def _check_validation(validation: float) -> None:
        """Raise an error for a validation share outside `[0, 1)`.

        Args:
            validation:
                The share of the samples of every series held out at its end.

        Raises:
            AnalysisError:
                The share is outside `[0, 1)`.
        """
        if not 0.0 <= validation < 1.0:
            error_msg = f'The validation share must lie in [0, 1), but got {validation}.'
            raise AnalysisError(error_msg)

    @staticmethod
    def _check_horizon(horizon: int) -> None:
        """Raise an error for a forecast horizon that is not a nonnegative integer.

        Args:
            horizon:
                The number of samples to forecast.

        Raises:
            AnalysisError:
                The horizon is not a nonnegative integer.
        """
        if isinstance(horizon, bool) or not isinstance(horizon, int | np.integer) or horizon < 0:
            error_msg = f'The horizon must be a nonnegative number of samples, but got {horizon!r}.'
            raise AnalysisError(error_msg)

    @staticmethod
    def _split_samples(series: list[Array], validation: float) -> tuple[list[Array], list[Array]]:
        """Return the leading training part and the trailing validation part of every series.

        Args:
            series:
                The values of every series.

            validation:
                The share of the samples of every series held out at its end, from 0 to below 1.

        Returns:
            parts:
                The training parts and the validation parts, each validation part starting at the last training sample.

        Raises:
            AnalysisError:
                The share is outside `[0, 1)`.
        """
        SparseIdentificationAnalysis._check_validation(validation)
        cuts = [len(values) - math.floor(validation * (len(values) - 1)) for values in series]
        return [values[:cut] for values, cut in zip(series, cuts, strict=True)], [
            values[cut - 1 :] for values, cut in zip(series, cuts, strict=True)
        ]

    @staticmethod
    def _relative_error(predicted: Sequence[Array], actual: Sequence[Array]) -> float:
        """Return the relative error `||predicted - actual|| / ||actual||` over every row of every array.

        Args:
            predicted:
                The predicted increments or derivatives.

            actual:
                The measured increments or derivatives.

        Returns:
            error:
                The relative error, or NaN when there is no row.
        """
        rows = [np.asarray(item, dtype=float) for item in actual if len(item)]
        if not rows:
            return math.nan
        estimate = np.concatenate([np.asarray(item, dtype=float) for item in predicted if len(item)])
        measured = np.concatenate(rows)
        return float(np.linalg.norm(estimate - measured) / np.linalg.norm(measured))

    @staticmethod
    def _state_names(labels: Sequence[str]) -> list[str]:
        """Return state names for variables: the last part of every dotted label, or the whole label where those clash.

        Args:
            labels:
                The names of the variables, such as `lorenz.x` or `cells.n[0]`.

        Returns:
            names:
                Valid and distinct identifiers, one per variable.
        """
        short = [SparseIdentificationAnalysis._identifier(label.rpartition('.')[2]) for label in labels]
        if len(set(short)) == len(short):
            return short
        full = [SparseIdentificationAnalysis._identifier(label) for label in labels]
        return full if len(set(full)) == len(full) else [f'{name}_{index}' for index, name in enumerate(full)]

    @staticmethod
    def _surrogate_system(
        kind: str,
        label: str,
        initial: Mapping[str, float],
        laws: Callable[[SimpleNamespace], list[BaseLaw]],
        time: BaseTime,
    ) -> System:
        """Return a system of one surrogate element with the given states, laws, and time.

        Args:
            kind:
                The name of the element type, such as `LinearSurrogate`.

            label:
                The label of the element.

            initial:
                The initial value of every state, by state name.

            laws:
                The function that returns the laws from the namespace of the element.

            time:
                The time of the system.

        Returns:
            system:
                The system.
        """
        states = [
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=value,
                annotation=Annotated[float, StateSpec()],
            )
            for name, value in initial.items()
        ]

        def initialize(self: Element, label: str, **values: float) -> None:
            Element.__init__(self, label)
            for name, value in {**initial, **values}.items():
                setattr(self, name, value)

        first = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in ('self', 'label')]
        vars(initialize)['__signature__'] = inspect.Signature([*first, *states])
        element_type = type(kind, (Element,), {'__init__': initialize, '_get_laws': lambda self, s: laws(s)})
        return System(elements=[element_type(label)], time=time)

    @staticmethod
    def _symbols_of(s: SimpleNamespace, names: Sequence[str]) -> list[Expression]:
        """Return the symbols of the named states of a namespace.

        Args:
            s:
                The namespace of the element.

            names:
                The state names.

        Returns:
            symbols:
                The symbols, in order.
        """
        return [getattr(s, name) for name in names]
