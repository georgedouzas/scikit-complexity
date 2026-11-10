"""Random values a stochastic map draws at every step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy import stats

from ...errors import ValidationError
from ...specs import DrawSpec, NamedDrawSpec, RunStateSpec, SampledDrawSpec
from ...types import Array, Indices
from ._base import BaseKernel
from ._draw_recorder import DrawRecorder


class DrawKernel(BaseKernel):
    """A kernel of the random values a stochastic map draws at every step.

    They are stored after the parameters of an extended vector.

    Args:
        offset:
            The index of the first draw in the extended vector, the size of the parameter vector.

        normal:
            One flag per uniform or normal draw, true for a standard normal value and false for a uniform value in
            `[0, 1)`, in the order the draws are taken, by component path, draw name, and member index.

        sampled:
            The draws from `scipy.stats` distributions, stored after the uniform and normal values and taken after
            them.

        orders:
            The orders of the members that gather tables read through, each drawn again in place as a random
            permutation after the other draws.

        named:
            Every draw by name, as runs record and replay them, in the order of the extended vector.
    """

    values_: None

    def __init__(
        self,
        offset: int,
        normal: npt.NDArray[np.bool_],
        sampled: tuple[SampledDrawSpec, ...] = (),
        orders: tuple[Indices, ...] = (),
        named: tuple[NamedDrawSpec, ...] = (),
    ) -> None:
        self.offset = offset
        self.normal = normal
        self.sampled = sampled
        self.orders = orders
        self.named = named

    @property
    def size(self) -> int:
        """Number of values drawn at every step."""
        return int(self.normal.size) + sum(item.rows for item in self.sampled)

    def extend(self, p: Array) -> Array:
        """Return a parameter vector extended with room for the draws.

        Args:
            p:
                The parameter vector.

        Returns:
            extended:
                A new vector with the parameters first and zeros for the draws.
        """
        extended = np.zeros(self.offset + self.size)
        extended[: self.offset] = p[: self.offset]
        return extended

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Draw fresh values into the extended parameter vector of the state.

        Uniform and normal runs are drawn as they come. The arguments of draws from `scipy.stats` distributions read the
        time and the state vector of the state. The orders are drawn last, in place.
        """
        self.values_ = None
        state = self._get_state_of(inputs)
        q, rng, t, x = state.p, self._get_generator_of(state), state.t, state.x
        cuts = [0, *(np.flatnonzero(np.diff(self.normal)) + 1).tolist(), self.normal.size]
        for start, end in itertools.pairwise(cuts):
            if start == end:
                continue
            count = end - start
            values = rng.standard_normal(count) if self.normal[start] else rng.random(count)
            q[self.offset + start : self.offset + end] = values
        for item in self.sampled:
            q[item.start : item.start + item.rows] = item.sample(t, x, q, rng)
        for order in self.orders:
            order[:] = rng.permutation(order.size)

    def record(self, state: RunStateSpec, recorder: DrawRecorder) -> None:
        """Draw values into the extended parameter vector through a recorder.

        Every named draw samples from the distribution its arguments give at the state, and the recorder keeps the
        values, or forces the recorded ones of a replay. The vector then holds the uniform or normal value that the
        lowered law turns into the kept value. The orders are drawn last, in place.

        Args:
            state:
                The state, with the extended parameter vector and the generator.

            recorder:
                The recorder.

        Raises:
            ValidationError:
                A draw reads more than the states and parameters of its member, so it cannot be recorded.
        """
        q, rng, t, x = state.p, self._get_generator_of(state), state.t, state.x
        for item in self.named:
            if item.arguments is None:
                error_msg = (
                    f'The draw {item.label!r} has arguments that read more than the states and parameters of its '
                    'member, so runs cannot record or replay it.'
                )
                raise ValidationError(error_msg)
            arguments = item.arguments(t, x, q)
            sampled = self._sample_draw(item.draw, arguments, item.rows, rng)
            values = recorder.draw(item.label, item.draw, arguments, sampled)
            q[item.start : item.start + item.rows] = self._get_slot_values(item.draw, arguments, values)
        for order in self.orders:
            order[:] = rng.permutation(order.size)

    @staticmethod
    def _get_slot_values(draw: DrawSpec, arguments: Sequence[npt.ArrayLike], values: Array) -> Array:
        """Return the uniform or normal values the lowered law turns into values of a draw.

        A draw from `scipy.stats` keeps its values.
        """
        size = values.size
        columns = [np.broadcast_to(np.asarray(item, dtype=float), (size,)) for item in arguments]
        with np.errstate(divide='ignore', invalid='ignore'):
            if draw.kind == 'uniform':
                low, high = columns
                return np.where(high != low, (values - low) / (high - low), 0.0)
            if draw.kind == 'normal':
                mean, deviation = columns
                return np.where(deviation != 0, (values - mean) / deviation, 0.0)
        if draw.kind == 'bernoulli':
            return np.where(values != 0, 0.0, 1.0)
        if draw.kind == 'categorical':
            cumulative = np.cumsum(np.stack(columns, axis=-1), axis=-1)
            chosen = values.astype(int)
            rows = np.arange(size)
            lower = np.where(chosen > 0, cumulative[rows, np.maximum(chosen - 1, 0)], 0.0)
            return (lower + cumulative[rows, chosen]) / 2 / cumulative[:, -1]
        return values

    @staticmethod
    def _sample_draw(draw: DrawSpec, arguments: Sequence[npt.ArrayLike], size: int, rng: np.random.Generator) -> Array:
        """Return values of a draw for a number of members.

        The arguments of the draw hold one value or one per member.

        Args:
            draw:
                The draw.

            arguments:
                The values of the draw's arguments, numbers or arrays with one value per member.

            size:
                The number of values.

            rng:
                The random number generator.

        Returns:
            values:
                One value per member.
        """
        values = [np.broadcast_to(np.asarray(item, dtype=float), (size,)) for item in arguments]
        if draw.kind == 'uniform':
            return values[0] + (values[1] - values[0]) * rng.random(size)
        if draw.kind == 'normal':
            return values[0] + values[1] * rng.standard_normal(size)
        if draw.kind == 'bernoulli':
            return (rng.random(size) < values[0]).astype(float)
        if draw.kind == 'categorical':
            weights = np.stack(values, axis=-1)
            cumulative = np.cumsum(weights, axis=-1)
            chosen = rng.random(size)[:, np.newaxis] * cumulative[:, -1:]
            return np.sum(chosen >= cumulative[:, :-1], axis=-1).astype(float)
        distribution = getattr(stats, str(draw.distribution))
        sampled = distribution.rvs(**dict(zip(draw.keywords, values, strict=True)), size=size, random_state=rng)
        return np.asarray(sampled, dtype=float)
