"""Fit reservoir surrogates to series.

The surrogates are next-generation reservoirs and echo state networks with ridge readouts.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt

from ..components.laws import BaseLaw, UpdateLaw
from ..components.spaces import DiscreteTime
from ..components.systems import CompositeSystem
from ..engine._utils import vector
from ..errors import AnalysisError
from ..specs import AnalysisTags, CallSpec, SamplesSpec
from ..types import Array, DataKind
from ._base import _import, _read_samples
from ._sparse_identification import (
    _check_horizon,
    _check_validation,
    _relative_error,
    _split_samples,
    _state_names,
    _surrogate_system,
    _symbols_of,
)
from .types import ReservoirNodeProtocol, Runs

RESERVOIR_KINDS = ('next_generation', 'echo_state')
ECHO_STATE_WARMUP = 100

TAGS = AnalysisTags(DataKind.SERIES)


class _ReservoirStep:
    """Map of the variables and the memory of a reservoir to their next values.

    The next values are those one sample later.
    """

    def __init__(
        self,
        kind: str,
        node: ReservoirNodeProtocol,
        readout: ReservoirNodeProtocol,
        mean: Array,
        scale: Array,
    ) -> None:
        self.kind = kind
        self.node = node
        self.readout = readout
        self.mean = mean
        self.scale = scale

    def _set_memory(self, memory: Array) -> None:
        """Set the memory of the node."""
        if self.kind == 'next_generation':
            self.node.store = np.asarray(memory, dtype=float).reshape(self.node.store.shape)
            return
        self.node.state = {'out': np.asarray(memory, dtype=float).copy()}

    def memory(self) -> Array:
        """Return the memory of the node.

        It is its window of past inputs or the activities of its units.

        Returns:
            memory:
                The memory as a flat array.
        """
        if self.kind == 'next_generation':
            return np.asarray(self.node.store, dtype=float).ravel()
        return np.asarray(self.node.state['out'], dtype=float).ravel()

    def monomials(self) -> npt.NDArray[np.int_]:
        """Return the monomials of a next-generation reservoir.

        They are given as the linear features that every one multiplies.

        Returns:
            monomials:
                One row per monomial, with the positions of its factors among the linear features.
        """
        return np.asarray(self.node._monomial_idx, dtype=int)

    def initialize(self, variables: Array) -> None:
        """Initialize the node for inputs of the size of the variables.

        Its memory is cleared.

        Args:
            variables:
                The values of the variables at one sample.
        """
        if not self.node.initialized:
            self.node.initialize(variables[None, :])
        self._set_memory(np.zeros_like(self.memory()))

    def features(self, values: Array) -> Array:
        """Return the features of the node driven by the series of the variables.

        They start from its current memory.

        Args:
            values:
                One row per sample and one column per variable.

        Returns:
            features:
                One row per sample.
        """
        return np.asarray(self.node.run((values - self.mean) / self.scale), dtype=float)

    def increments(self, features: Array) -> Array:
        """Return the increments of the variables the readout predicts from features.

        Args:
            features:
                One row per sample.

        Returns:
            increments:
                One row per sample and one column per variable.
        """
        if not len(features):
            return np.zeros((0, len(self.scale)))
        return np.asarray(self.readout.run(features), dtype=float) * self.scale

    def evaluate(self, t: float, *values: float) -> Array:
        """Return the variables and the memory one sample later, applying the readout once.

        Args:
            t:
                The time, which the map does not read.

            *values:
                The variables, then the memory.

        Returns:
            values:
                The variables, then the memory, one sample later.
        """
        del t
        size = len(self.mean)
        variables = np.asarray(values[:size], dtype=float)
        self._set_memory(np.asarray(values[size:], dtype=float))
        features = np.asarray(self.node.step((variables - self.mean) / self.scale), dtype=np.float64)
        following = variables + np.asarray(self.readout.step(features), dtype=float) * self.scale
        return np.concatenate([following, self.memory()])


@dataclass(frozen=True)
class ReservoirResult:
    """Reservoir system whose ridge readout gives the increments of the variables.

    The increments are over one sample.

    Args:
        samples:
            The series of every run, the names of the variables, and the step.

        system:
            The system of one surrogate element under discrete time, one step per sample, whose states are the
            variables and then the memory of the reservoir.

        labels:
            The variables the surrogate was fitted to.

        states:
            The names of the states of the surrogate element, the variables first.

        last_state:
            The variables at the last sample of the last series and the memory the series left, where forecasts start.

        dt:
            The time between the samples it was fitted to, which one step of the system stands for.

        training_error:
            The relative error of the predicted increments on the training samples, with the reservoir driven by the
            series.

        validation_error:
            The relative error on the held-out samples, NaN when none are held out.

        step:
            The map from the states to the states one step later, which the update law calls, holding the fitted
            reservoir node `node`, the ridge node `readout`, and the standardisation `mean` and `scale`.

        forecast:
            The variables at the `horizon` samples after the last one, one row per sample.
    """

    samples: SamplesSpec
    system: CompositeSystem
    labels: list[str]
    states: list[str]
    last_state: Array
    dt: float
    training_error: float
    validation_error: float
    step: _ReservoirStep
    forecast: Array


def reservoir(
    source: Runs,
    *,
    variables: Sequence[str] | None = None,
    kind: str = 'next_generation',
    delay: int = 2,
    order: int = 2,
    units: int = 300,
    spectral_radius: float = 0.9,
    leak: float = 1.0,
    input_scaling: float = 0.5,
    ridge: float = 1e-4,
    standardize: bool = True,
    warmup: int | None = None,
    validation: float = 0.2,
    horizon: int = 0,
    dt: float = 1.0,
    random_state: int | None = 0,
    label: str = 'surrogate',
) -> ReservoirResult:
    """Return the reservoir system whose ridge readout gives the variable increments.

    The reservoir comes from ReservoirPy. The reservoir is driven by the variables, standardized unless `standardize` is
    false, and a ridge regression from its features to the increments of the variables over one sample is fitted on the
    training samples. A next-generation reservoir takes as features the variables at `delay` consecutive samples and
    their monomials of degree `order`. An echo state network takes the activities of `units` leaky tanh units with
    random recurrent weights. The fitted system runs under discrete time, one step per sample, and starts at the last
    sample of the last run with the memory the runs left, so a run of the system is the autonomous forecast.

    Args:
        source:
            A trajectory, an array with one row per sample, or several runs as a list or a 3-D array.

        variables:
            The variables of trajectories, or `None` for every scalar state or observable.

        kind:
            The reservoir, `'next_generation'` or `'echo_state'`.

        delay:
            The number of consecutive samples of a next-generation reservoir.

        order:
            The degree of the monomials of a next-generation reservoir.

        units:
            The number of units of an echo state network.

        spectral_radius:
            The spectral radius of the recurrent weights of an echo state network.

        leak:
            The leak rate of the units of an echo state network, 1 for no leak.

        input_scaling:
            The scale of the input weights of an echo state network.

        ridge:
            The strength of the ridge penalty of the readout.

        standardize:
            Whether the reservoir reads the variables divided by their standard deviations after removing their means,
            or the variables as they are.

        warmup:
            The number of first samples of every series left out of the fit while the memory fills, or `None` for
            `delay` with a next-generation reservoir and 100 with an echo state network.

        validation:
            The share of the samples of every run held out at its end to measure the validation error.

        horizon:
            The number of samples to forecast after the last sample of the last run.

        dt:
            The time between the rows of an array.

        random_state:
            The seed of the random weights of an echo state network.

        label:
            The label of the surrogate element.

    Returns:
        result:
            The fitted system, its errors, its reservoir map, and the forecast.

    Raises:
        AnalysisError:
            The reservoir kind is unknown, the validation share lies outside `[0, 1)`, the horizon is negative, or no
            training sample is left after the warmup.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import reservoir
        >>> times = np.arange(400.0)
        >>> result = reservoir(np.sin(0.1 * times), variables=['x'], horizon=10)
        >>> result.states
        ['x', 'memory_0', 'memory_1']
        >>> bool(abs(result.forecast[-1, 0] - np.sin(0.1 * 409)) < 0.01)
        True
    """
    _check_validation(validation)
    _check_horizon(horizon)
    if kind not in RESERVOIR_KINDS:
        error_msg = f'The reservoir kind {kind!r} is unknown. Choose one of {RESERVOIR_KINDS}.'
        raise AnalysisError(error_msg)
    samples = _read_samples(source, variables, dt)
    node = _reservoir_node(
        kind,
        delay=delay,
        order=order,
        units=units,
        spectral_radius=spectral_radius,
        leak=leak,
        input_scaling=input_scaling,
        random_state=random_state,
    )
    training, _ = _split_samples(samples.series, validation)
    joined = np.concatenate(training)
    deviation = joined.std(axis=0)
    step = _ReservoirStep(
        kind,
        node,
        _import('reservoirpy.nodes').Ridge(ridge=ridge),
        joined.mean(axis=0) if standardize else np.zeros(joined.shape[1]),
        np.where(deviation > 0, deviation, 1.0) if standardize else np.ones(joined.shape[1]),
    )
    default_warmup = delay if kind == 'next_generation' else ECHO_STATE_WARMUP
    skipped = default_warmup if warmup is None else warmup
    training_error, validation_error = _fit_readout(step, samples.series, training, skipped)
    memory = step.memory()
    states = _state_names([*samples.labels, *(f'memory[{index}]' for index in range(len(memory)))])
    last_state = np.asarray([*samples.series[-1][-1], *memory], dtype=float)
    system = _surrogate_system(
        label,
        dict(zip(states, map(float, last_state), strict=True)),
        _reservoir_laws(states, step),
        DiscreteTime(),
    )
    current = last_state
    rows = []
    for _ in range(horizon):
        current = step.evaluate(0.0, *current)
        rows.append(current[: len(samples.labels)])
    return ReservoirResult(
        samples,
        system,
        samples.labels,
        states,
        last_state,
        samples.dt,
        training_error,
        validation_error,
        step,
        np.asarray(rows).reshape(horizon, len(samples.labels)),
    )


def _reservoir_node(
    kind: str,
    *,
    delay: int,
    order: int,
    units: int,
    spectral_radius: float,
    leak: float,
    input_scaling: float,
    random_state: int | None,
) -> ReservoirNodeProtocol:
    """Return the ReservoirPy node of a kind of reservoir.

    The kinds are a nonlinear vector autoregression or a unit reservoir.
    """
    nodes = _import('reservoirpy.nodes')
    if kind == 'next_generation':
        return nodes.NVAR(delay=delay, order=order, strides=1)
    return nodes.Reservoir(
        units=units,
        sr=spectral_radius,
        lr=leak,
        input_scaling=input_scaling,
        seed=random_state,
    )


def _fit_readout(
    step: _ReservoirStep,
    series: list[Array],
    training: list[Array],
    skipped: int,
) -> tuple[float, float]:
    """Fit the readout of a reservoir driven by every series, and return its errors.

    The errors are the training and validation errors.
    """
    features = []
    for values in series:
        step.initialize(values[0])
        features.append(step.features(values[:-1]))
    cuts = [len(values) - 1 for values in training]
    inputs = [found[skipped:cut] for found, cut in zip(features, cuts, strict=True)]
    targets = [np.diff(values, axis=0)[skipped:] for values in training]
    if not sum(len(item) for item in inputs):
        error_msg = f'No training sample is left after the warmup of {skipped} samples. Give longer series.'
        raise AnalysisError(error_msg)
    step.readout.fit(np.concatenate(inputs), np.concatenate(targets) / step.scale)
    increments = [np.diff(values, axis=0) for values in series]
    return _relative_error(
        [step.increments(item) for item in inputs],
        targets,
    ), _relative_error(
        [step.increments(found[cut:]) for found, cut in zip(features, cuts, strict=True)],
        [found[cut:] for found, cut in zip(increments, cuts, strict=True)],
    )


def _reservoir_laws(names: Sequence[str], step: _ReservoirStep) -> Callable[[SimpleNamespace], list[BaseLaw]]:
    """Return the function that gives the one update law of every state.

    It calls the reservoir map once.
    """

    def laws(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the update law."""
        symbols = _symbols_of(s, names)
        return [UpdateLaw(vector(symbols), CallSpec(step.evaluate, tuple(symbols)))]

    return laws
