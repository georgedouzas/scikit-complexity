"""Random values a stochastic map draws at every step, stored after the parameters of an extended vector."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...types import Array
from ._sampled_draw import SampledDraw


@dataclass(frozen=True)
class Draws:
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

    offset: int
    normal: npt.NDArray[np.bool_]
    sampled: tuple[SampledDraw, ...] = ()

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

    def fill(self, q: Array, rng: np.random.Generator, t: float = 0.0, x: Array | None = None) -> None:
        """Draw fresh values into an extended parameter vector, in order, uniform and normal runs as they come.

        Args:
            q:
                The extended parameter vector, changed in place.

            rng:
                The random number generator.

            t:
                The time, which the arguments of draws from `scipy.stats` distributions may read.

            x:
                The state vector, which the arguments of draws from `scipy.stats` distributions may read.
        """
        cuts = [0, *(np.flatnonzero(np.diff(self.normal)) + 1).tolist(), self.normal.size]
        for start, end in itertools.pairwise(cuts):
            if start == end:
                continue
            count = end - start
            values = rng.standard_normal(count) if self.normal[start] else rng.random(count)
            q[self.offset + start : self.offset + end] = values
        state = np.zeros(0) if x is None else x
        for item in self.sampled:
            q[item.start : item.start + item.rows] = item.sample(t, state, q, rng)
