"""Fit reservoir surrogates to series: next-generation reservoirs and echo state networks with ridge readouts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt

from ..core.errors import AnalysisError
from ..core.laws import BaseLaw, UpdateLaw
from ..core.laws.utils import Call
from ..core.spaces import DiscreteTime
from ..engine.problems import vector
from ._lazy import module
from ._surrogate_systems import (
    BaseSurrogate,
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
    return nodes.Reservoir(
        units=units,
        sr=spectral_radius,
        lr=leak,
        input_scaling=input_scaling,
        seed=random_state,
    )


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


def _reservoir_laws(names: Sequence[str], step: _ReservoirStep) -> Callable[[SimpleNamespace], list[BaseLaw]]:
    """Return the function that gives the one update law of every state, which calls the reservoir map once."""

    def laws(s: SimpleNamespace) -> list[BaseLaw]:
        """Return the update law."""
        symbols = symbols_of(s, names)
        return [UpdateLaw(vector(symbols), Call(step.evaluate, tuple(symbols)))]

    return laws


class ReservoirSurrogate(BaseSurrogate):
    """Reservoir surrogate whose ridge readout gives the increments of the variables over one sample, from ReservoirPy.

    The reservoir is driven by the standardized variables, and a ridge regression from its features to the increments
    of the variables over one sample is fitted on the training samples. A next-generation reservoir takes as features
    the variables at `delay` consecutive samples and their monomials of degree `order`. An echo state network takes
    the activities of `units` leaky tanh units with random recurrent weights. The surrogate runs under discrete time,
    one step per sample, and starts at the last sample of the last series with the memory the series left, so a run of
    the system is the autonomous forecast.

    Args:
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

    Attributes:
        system_:
            The system of one surrogate element under discrete time, one step per sample, whose states are the
            variables and then the memory of the reservoir.

        labels_:
            The variables the surrogate was fitted to.

        states_:
            The names of the states of the surrogate element, the variables first.

        last_state_:
            The variables at the last sample of the last series and the memory the series left, where forecasts start.

        dt_:
            The time between the samples it was fitted to, which one step of the system stands for.

        training_error_:
            The relative error of the predicted increments on the training samples, with the reservoir driven by the
            series.

        validation_error_:
            The relative error on the held-out samples, NaN when none are held out.

        step_:
            The map from the states to the states one step later, which the update law calls.

    Raises:
        AnalysisError:
            The reservoir kind is unknown, or no training sample is left after the warmup.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import ReservoirSurrogate, Series
        >>> times = np.arange(400.0)
        >>> surrogate = ReservoirSurrogate().prepare(Series(times, np.sin(0.1 * times)[:, None], ['x']))
        >>> surrogate.states_
        ['x', 'memory_0', 'memory_1']
        >>> bool(abs(surrogate.run(10)[-1, 0] - np.sin(0.1 * 409)) < 0.01)
        True
    """

    step_: _ReservoirStep

    def __init__(
        self,
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
    ) -> None:
        self.names = names
        self.kind = kind
        self.delay = delay
        self.order = order
        self.units = units
        self.spectral_radius = spectral_radius
        self.leak = leak
        self.input_scaling = input_scaling
        self.ridge = ridge
        self.warmup = warmup
        self.validation = validation
        self.random_state = random_state
        self.label = label

    def _check(self) -> None:
        """Raise an error for an unknown reservoir kind or a validation share outside `[0, 1)`."""
        super()._check()
        if self.kind not in RESERVOIR_KINDS:
            error_msg = f'The reservoir kind {self.kind!r} is unknown. Choose one of {RESERVOIR_KINDS}.'
            raise AnalysisError(error_msg)

    def _prepare(self, data: SurrogateData) -> None:
        """Drive the reservoir by the series, fit its readout, and set the surrogate system and its errors."""
        samples = samples_of(data, self.names)
        node = _reservoir_node(
            self.kind,
            delay=self.delay,
            order=self.order,
            units=self.units,
            spectral_radius=self.spectral_radius,
            leak=self.leak,
            input_scaling=self.input_scaling,
            random_state=self.random_state,
        )
        training, _ = split_samples(samples.series, self.validation)
        joined = np.concatenate(training)
        step = _ReservoirStep(
            self.kind,
            node,
            module('reservoirpy.nodes').Ridge(ridge=self.ridge),
            joined.mean(axis=0),
            np.where(joined.std(axis=0) > 0, joined.std(axis=0), 1.0),
        )
        default_warmup = self.delay if self.kind == 'next_generation' else ECHO_STATE_WARMUP
        skipped = default_warmup if self.warmup is None else self.warmup
        self.training_error_, self.validation_error_ = _fit_readout(step, samples.series, training, skipped)
        memory = step.memory()
        states = state_names([*samples.labels, *(f'memory[{index}]' for index in range(len(memory)))])
        self.last_state_ = np.asarray([*samples.series[-1][-1], *memory], dtype=float)
        self.system_ = surrogate_system(
            'ReservoirSurrogate',
            self.label,
            dict(zip(states, map(float, self.last_state_), strict=True)),
            _reservoir_laws(states, step),
            DiscreteTime(),
        )
        self.labels_ = samples.labels
        self.states_ = states
        self.dt_ = samples.dt
        self.step_ = step

    def _run(self, steps: int) -> Array:
        """Return the autonomous forecast of the variables, applying the map once per sample after the last one."""
        current = self.last_state_
        rows = []
        for _ in range(steps):
            current = self.step_.evaluate(0.0, *current)
            rows.append(current[: len(self.labels_)])
        return np.asarray(rows).reshape(steps, len(self.labels_))
