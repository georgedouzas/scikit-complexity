"""Random values a stochastic map draws at every step, stored after the parameters of an extended vector."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy import stats

from ...specs import DrawSpec, RunStateSpec, SampledDrawSpec
from ...types import Array
from ._base import BaseKernel


class DrawKernel(BaseKernel):
    """Random values a stochastic map draws at every step, stored after the parameters of an extended vector.

    Args:
        offset:
            The index of the first draw in the extended vector, the size of the parameter vector.

        normal:
            One flag per uniform or normal draw, true for a standard normal value and false for a uniform value in
            `[0, 1)`, in the order the draws are taken, by component path, draw name, and member index.

        sampled:
            The draws from `scipy.stats` distributions, stored after the uniform and normal values and taken after
            them.
    """

    values_: None

    def __init__(self, offset: int, normal: npt.NDArray[np.bool_], sampled: tuple[SampledDrawSpec, ...] = ()) -> None:
        self.offset = offset
        self.normal = normal
        self.sampled = sampled

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
        """Draw fresh values into the extended parameter vector of the state, uniform and normal runs as they come.

        The arguments of draws from `scipy.stats` distributions read the time and the state vector of the state.
        """
        self.values_ = None
        state = self._state_of(inputs)
        q, rng, t, x = state.p, self._generator_of(state), state.t, state.x
        cuts = [0, *(np.flatnonzero(np.diff(self.normal)) + 1).tolist(), self.normal.size]
        for start, end in itertools.pairwise(cuts):
            if start == end:
                continue
            count = end - start
            values = rng.standard_normal(count) if self.normal[start] else rng.random(count)
            q[self.offset + start : self.offset + end] = values
        for item in self.sampled:
            q[item.start : item.start + item.rows] = item.sample(t, x, q, rng)

    @staticmethod
    def _sample_draw(draw: DrawSpec, arguments: Sequence[npt.ArrayLike], size: int, rng: np.random.Generator) -> Array:
        """Return values of a draw for a number of members, with arguments that hold one value or one per member.

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
