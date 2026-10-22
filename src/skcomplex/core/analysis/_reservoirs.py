"""Fit reservoir surrogates to series: next-generation reservoirs and echo state networks with ridge readouts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ..base import AnalysisError, vector
from ..laws import Call, Law, Symbols, Update
from ..space import DiscreteTime
from ._kinds import DataKind, analysis
from ._lazy import module
from ._surrogates import (
    Surrogate,
    SurrogateData,
    relative_error,
    samples_of,
    split_samples,
    state_names,
    surrogate_system,
    symbols_of,
)

type Array = npt.NDArray[np.float64]
RESERVOIR_KINDS = ('next_generation', 'echo_state')
ECHO_STATE_WARMUP = 100


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
) -> Any:  # noqa: ANN401
    """Return the ReservoirPy node of a kind of reservoir, a nonlinear vector autoregression or a reservoir of units."""
    nodes = module('reservoirpy.nodes')
    if kind == 'next_generation':
        return nodes.NVAR(delay=delay, order=order, strides=1)
    if kind == 'echo_state':
        return nodes.Reservoir(
            units=units,
            sr=spectral_radius,
            lr=leak,
            input_scaling=input_scaling,
            seed=random_state,
        )
    error_msg = f'The reservoir kind {kind!r} is unknown. Choose one of {RESERVOIR_KINDS}.'
    raise AnalysisError(error_msg)


class _ReservoirStep:
    """Map from the variables and the memory of a reservoir to their values one sample later."""

    def __init__(self, kind: str, node: Any, readout: Any, mean: Array, scale: Array) -> None:  # noqa: ANN401
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
        """Return the memory of the node, its window of past inputs or the activities of its units.

        Returns:
            memory:
                The memory as a flat array.
        """
        if self.kind == 'next_generation':
            return np.asarray(self.node.store, dtype=float).ravel()
        return np.asarray(self.node.state['out'], dtype=float).ravel()

    def initialize(self, variables: Array) -> None:
        """Initialize the node for inputs of the size of the variables and clear its memory.

        Args:
            variables:
                The values of the variables at one sample.
        """
        if not self.node.initialized:
            self.node.initialize(variables[None, :])
        self._set_memory(np.zeros_like(self.memory()))

    def features(self, values: Array) -> Array:
        """Return the features of the node driven by the series of the variables, from its current memory.

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

    def __call__(self, t: float, *values: float) -> Array:
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
        features = self.node.step((variables - self.mean) / self.scale)
        following = variables + np.asarray(self.readout.step(features), dtype=float) * self.scale
        return np.concatenate([following, self.memory()])


def _fit_readout(step: _ReservoirStep, series: list[Array], training: list[Array], skipped: int) -> tuple[float, float]:
    """Fit the readout of a reservoir driven by every series, and return its training and validation errors."""
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
    return relative_error([step.increments(item) for item in inputs], targets), relative_error(
        [step.increments(found[cut:]) for found, cut in zip(features, cuts, strict=True)],
        [found[cut:] for found, cut in zip(increments, cuts, strict=True)],
    )


def _reservoir_laws(names: Sequence[str], step: _ReservoirStep) -> Callable[[Symbols], list[Law]]:
    """Return the function that gives the one update law of every state, which calls the reservoir map once."""

    def laws(s: Symbols) -> list[Law]:
        """Return the update law."""
        symbols = symbols_of(s, names)
        return [Update(vector(symbols), Call(step, *symbols))]

    return laws


@dataclass(frozen=True)
class ReservoirModel(Surrogate):
    """Reservoir surrogate whose readout gives the increments of the variables over one sample.

    Args:
        system:
            The system of one surrogate element under discrete time, one step per sample, whose states are the
            variables and then the memory of the reservoir.

        labels:
            The variables the surrogate was fitted to.

        states:
            The names of the states of the surrogate element, the variables first.

        dt:
            The time between the samples it was fitted to, which one step of the system stands for.

        training_error:
            The relative error of the predicted increments on the training samples, with the reservoir driven by the
            series.

        validation_error:
            The relative error on the held-out samples, NaN when none are held out.

        step:
            The map from the states to the states one step later, which the update law calls.
    """

    step: Callable[..., Array]

    def forecast(self, steps: int) -> Array:
        """Return the autonomous forecast of the variables, from the last sample of the last series.

        Args:
            steps:
                The number of steps.

        Returns:
            variables:
                One row per step, the last sample first, and one column per variable.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.core.analysis import Series, fit_reservoir
            >>> times = np.arange(400.0)
            >>> model = fit_reservoir(Series(times, np.sin(0.1 * times)[:, None], ['x']))
            >>> bool(abs(model.forecast(10)[-1, 0] - np.sin(0.1 * 409)) < 0.01)
            True
        """
        element = self.system.elements[0]
        current = np.asarray([getattr(element, name) for name in self.states], dtype=float)
        rows = [current]
        for _ in range(steps):
            current = self.step(0.0, *current)
            rows.append(current)
        return np.asarray(rows)[:, : len(self.labels)]


@analysis(DataKind.SERIES)
def fit_reservoir(
    data: SurrogateData,
    names: Sequence[str] | None = None,
    *,
    kind: str = 'next_generation',
    delay: int = 2,
    order: int = 2,
    units: int = 300,
    spectral_radius: float = 0.9,
    leak: float = 1.0,
    input_scaling: float = 0.5,
    ridge: float = 1e-4,
    warmup: int | None = None,
    validation: float = 0.2,
    random_state: int | None = 0,
    label: str = 'surrogate',
) -> ReservoirModel:
    """Return the reservoir surrogate fitted to series, with a reservoir and a ridge readout from ReservoirPy.

    The reservoir is driven by the standardized variables, and a ridge regression from its features to the increments
    of the variables over one sample is fitted on the training samples. A next-generation reservoir takes as features
    the variables at `delay` consecutive samples and their monomials of degree `order`. An echo state network takes
    the activities of `units` leaky tanh units with random recurrent weights. The surrogate runs under discrete time,
    one step per sample, and starts at the last sample of the last series with the memory the series left.

    Args:
        data:
            A trajectory, a series, or a sequence of them, sampled at one uniform step.

        names:
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

        warmup:
            The number of first samples of every series left out of the fit while the memory fills, or `None` for
            `delay` with a next-generation reservoir and 100 with an echo state network.

        validation:
            The share of the samples of every series held out at its end to measure the validation error.

        random_state:
            The seed of the random weights of an echo state network.

        label:
            The label of the surrogate element.

    Returns:
        model:
            The surrogate system, its errors, and its map over one step.

    Raises:
        AnalysisError:
            The reservoir kind is unknown, or no training sample is left after the warmup.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import Series, fit_reservoir
        >>> times = np.arange(400.0)
        >>> model = fit_reservoir(Series(times, np.sin(0.1 * times)[:, None], ['x']))
        >>> model.states
        ['x', 'memory_0', 'memory_1']
    """
    samples = samples_of(data, names)
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
    training, _ = split_samples(samples.series, validation)
    joined = np.concatenate(training)
    step = _ReservoirStep(
        kind,
        node,
        module('reservoirpy.nodes').Ridge(ridge=ridge),
        joined.mean(axis=0),
        np.where(joined.std(axis=0) > 0, joined.std(axis=0), 1.0),
    )
    skipped = (delay if kind == 'next_generation' else ECHO_STATE_WARMUP) if warmup is None else warmup
    errors = _fit_readout(step, samples.series, training, skipped)
    memory = step.memory()
    states = state_names([*samples.labels, *(f'memory[{index}]' for index in range(len(memory)))])
    system = surrogate_system(
        'ReservoirSurrogate',
        label,
        dict(zip(states, map(float, [*samples.series[-1][-1], *memory]), strict=True)),
        _reservoir_laws(states, step),
        DiscreteTime(),
    )
    return ReservoirModel(
        system,
        samples.labels,
        states,
        samples.dt,
        *errors,
        step,
    )
