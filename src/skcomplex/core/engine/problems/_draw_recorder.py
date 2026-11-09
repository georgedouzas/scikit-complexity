"""Record the named draws and jump firings of a run, or replay recorded ones."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy import stats

from ...base import BaseConfig
from ...errors import ValidationError
from ...specs import DrawLogSpec, DrawRecordSpec, DrawSpec
from ...types import Array

KEYWORDS = {'uniform': ('low', 'high'), 'normal': ('mean', 'deviation'), 'bernoulli': ('p',)}


class DrawRecorder(BaseConfig[None, int]):
    """Recorder of the named draws and jump firings of a run, which also replays them.

    A run that records keeps every value it draws by name and step, with the arguments of its distribution, and every
    firing of a jump law with its time. A run that replays a log takes every named draw and every firing from it
    instead, while it computes the arguments of each distribution from its own parameters and states, and sums the
    log-probability of the forced values. `prepare()` forgets earlier draws and `run(step)` moves the recorder to the
    step of the run the next draws fall in. The places that draw, such as the draws of stochastic maps, the named draws
    of agent rules, and `GillespieSolver`, call `draw`, `firings`, and `record_firing`.

    Args:
        replay:
            The log to replay, or `None` to record.

    Attributes:
        step_:
            The step of the run the next draws fall in.

        log_probability_:
            The total log-probability of the draws and firings so far.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.problems._draw_recorder import DrawRecorder
        >>> from skcomplex.core.specs import DrawSpec
        >>> recorder = DrawRecorder().prepare().run(1)
        >>> recorder.draw('walker.kick', DrawSpec('normal', 'kick'), [0.0, 1.0], np.array([0.5])).tolist()
        [0.5]
        >>> round(recorder.log_probability_, 4)
        -1.0439
    """

    step_: int
    log_probability_: float

    def __init__(self, replay: DrawLogSpec | None = None) -> None:
        self.replay = replay

    def _prepare(self, context: None) -> None:
        """Forget earlier draws and firings and check the log to replay."""
        if self.replay is not None:
            self.replay._check()
        self.step_ = 0
        self.log_probability_ = 0.0
        self._kinds: dict[str, DrawSpec] = {}
        self._steps: dict[str, list[Array]] = defaultdict(list)
        self._values: dict[str, list[Array]] = defaultdict(list)
        self._arguments: dict[str, list[dict[str, Array]]] = defaultdict(list)
        self._cursors: dict[str, int] = defaultdict(int)
        self._firings: list[tuple[float, int]] = []
        self._firing_cursor = 0

    def _run(self, inputs: int | None) -> None:
        """Move the recorder to the step the next draws fall in."""
        self.step_ = 0 if inputs is None else int(inputs)

    @property
    def replaying(self) -> bool:
        """Whether the recorder replays a log."""
        return self.replay is not None

    def _get_forced(self, label: str, size: int) -> Array:
        """Return the recorded values of the next draw of a name.

        Args:
            label:
                The name of the draw, the path of its owner and its name.

            size:
                The number of values the draw takes.

        Returns:
            values:
                The recorded values.

        Raises:
            ValidationError:
                The log has no draws of the name, or its next values do not belong to the current step.
        """
        replay = self.replay
        if replay is None:
            error_msg = 'The recorder records, so it forces no draws.'
            raise ValidationError(error_msg)
        record = replay.draws.get(label)
        if record is None:
            error_msg = (
                f'The system draws {label!r}, which the replayed trajectory lacks. The trajectory holds the draws '
                f'{", ".join(sorted(replay.draws)) or "none"}.'
            )
            raise ValidationError(error_msg)
        start = self._cursors[label]
        steps = np.asarray(record.steps[start : start + size])
        if steps.size < size or np.any(steps != self.step_):
            error_msg = (
                f'At step {self.step_} the system draws {size} values of {label!r}, but the replayed trajectory holds '
                f'{int(np.sum(steps == self.step_))} at that step. Replay a trajectory of a run with the same draws.'
            )
            raise ValidationError(error_msg)
        self._cursors[label] = start + size
        return np.asarray(record.values[start : start + size], dtype=float)

    @staticmethod
    def _get_keywords(draw: DrawSpec) -> tuple[str, ...]:
        """Return the keywords of the arguments of a draw."""
        if draw.kind == 'categorical':
            return ('weights',)
        return KEYWORDS.get(draw.kind, draw.keywords)

    @staticmethod
    def _get_named(draw: DrawSpec, arguments: Sequence[npt.ArrayLike], size: int) -> dict[str, Array]:
        """Return the arguments of a draw by keyword, one value or row per drawn value."""
        columns = [np.broadcast_to(np.asarray(item, dtype=float), (size,)) for item in arguments]
        if draw.kind == 'categorical':
            return {'weights': np.stack(columns, axis=-1)}
        return dict(zip(DrawRecorder._get_keywords(draw), columns, strict=True))

    @staticmethod
    def _log_probability(draw: DrawSpec, named: dict[str, Array], values: Array) -> float:
        """Return the total log-probability of values under the arguments of a draw."""
        with np.errstate(divide='ignore', invalid='ignore'):
            if draw.kind == 'uniform':
                logs = stats.uniform.logpdf(values, named['low'], named['high'] - named['low'])
            elif draw.kind == 'normal':
                logs = stats.norm.logpdf(values, named['mean'], named['deviation'])
            elif draw.kind == 'bernoulli':
                logs = stats.bernoulli.logpmf(values, np.clip(named['p'], 0.0, 1.0))
            elif draw.kind == 'categorical':
                weights = named['weights']
                chosen = weights[np.arange(values.size), values.astype(int)]
                logs = np.log(chosen / weights.sum(axis=-1))
            else:
                distribution = getattr(stats, str(draw.distribution))
                density = distribution.logpmf if isinstance(distribution, stats.rv_discrete) else distribution.logpdf
                logs = density(values, **named)
        return float(np.sum(logs))

    def draw(self, label: str, draw: DrawSpec, arguments: Sequence[npt.ArrayLike], values: Array) -> Array:
        """Record the values of a named draw, or return the recorded ones of a replay.

        The caller samples the values first, so a replay takes the same random numbers from the generator as the
        recorded run and every draw that is not named, such as the order of agents, stays the same.

        Args:
            label:
                The path of the owner and the name of the draw, such as `walker.kick`.

            draw:
                The draw, with its kind and distribution.

            arguments:
                The values of the arguments of the distribution, in the order of the draw, numbers or arrays with one
                value per drawn value.

            values:
                The freshly sampled values.

        Returns:
            values:
                The sampled values, or the recorded ones of a replay.

        Raises:
            ValidationError:
                A replay has no recorded values of the draw at the current step.
        """
        size = int(np.size(values))
        if self.replaying:
            values = self._get_forced(label, size)
        named = self._get_named(draw, arguments, size)
        self.log_probability_ += self._log_probability(draw, named, np.asarray(values, dtype=float))
        self._kinds[label] = draw
        self._steps[label].append(np.full(size, self.step_))
        self._values[label].append(np.asarray(values, dtype=float).reshape(size))
        self._arguments[label].append(named)
        return np.asarray(values, dtype=float)

    def get_firings(self, t: float, until: float) -> list[tuple[float, int]]:
        """Return the recorded firings after a time and up to another, in order.

        Args:
            t:
                The time after which the firings fall.

            until:
                The time up to which they fall.

        Returns:
            firings:
                The time and the law of every firing.
        """
        replay = self.replay
        if replay is None:
            return []
        times, laws = replay.firing_times, replay.firing_laws
        found = []
        while self._firing_cursor < len(times) and times[self._firing_cursor] <= until:
            time = float(times[self._firing_cursor])
            if time > t:
                found.append((time, int(laws[self._firing_cursor])))
            self._firing_cursor += 1
        return found

    def record_firing(self, time: float, law: int, log_probability: float) -> None:
        """Record a firing of a jump law with the log-probability it adds.

        Args:
            time:
                The time of the firing.

            law:
                The index of the jump law.

            log_probability:
                The log-probability the firing and the waiting before it add.
        """
        self._firings.append((time, law))
        self.add_log_probability(log_probability)

    def add_log_probability(self, value: float) -> None:
        """Add a log-probability to the total.

        Args:
            value:
                The log-probability.
        """
        self.log_probability_ += value

    @property
    def log_(self) -> DrawLogSpec:
        """Log of the draws and firings so far."""
        draws = {}
        for label, draw in self._kinds.items():
            keywords = self._get_keywords(draw)
            arguments = {
                keyword: np.concatenate([item[keyword] for item in self._arguments[label]]) for keyword in keywords
            }
            draws[label] = DrawRecordSpec(
                draw.kind,
                draw.distribution,
                np.concatenate(self._steps[label]),
                np.concatenate(self._values[label]),
                arguments,
            )
        times = np.asarray([time for time, _ in self._firings], dtype=float)
        laws = np.asarray([law for _, law in self._firings], dtype=np.int64)
        return DrawLogSpec(draws, times, laws, self.log_probability_)
