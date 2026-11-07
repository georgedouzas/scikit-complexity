"""Update of a channel of a Lenia world by the growth maps of the convolutions of channels with ring kernels."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, exp, sqrt

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import both, clip, convolve, displacement, where
from ....core.specs import RoleSpec
from ....core.types import Array

type RingKernel = tuple[str, float, float, Sequence[float], float, float]


class LeniaUpdateInteraction(Interaction):
    """Update of a channel of a Lenia world by the growth maps of the convolutions of channels with ring kernels.

    Every kernel `(source, radius, h, beta, mu, sigma)` reads the state of its source channel, and its concentric
    rings peak at `beta`. At the distance `q` in units of its radius, with `B` rings, it is `beta_i K_C(B q - i)` on
    the ring `i = floor(B q)`, with the bump `K_C(x) = exp(4 - 1 / (x (1 - x)))`, normalised to a sum of 1 over the
    sites. Its growth map is `G(u) = 2 exp(-(u - mu)^2 / (2 sigma^2)) - 1`, and one step adds the growths weighted by
    `h` over the sum of the weights and clips the channel, `A <- [A + sum_k (h_k / h) G_k(K_k * A_k) / T]` in
    `[0, 1]`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the world.

        channel:
            The name of the state of the channel that the update changes.

        kernels:
            The kernels `(source, radius, h, beta, mu, sigma)`, with the name of the state of the source channel and
            the radius in sites.

        T:
            The number of steps per unit time.

        n_dim:
            The number of dimensions of the world.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LeniaUpdateInteraction
        >>> LeniaUpdateInteraction('update', target='world').T
        10.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        channel: str = 'A',
        kernels: Sequence[RingKernel] = (('A', 13.0, 1.0, (1.0,), 0.15, 0.015),),
        T: float = 10.0,
        n_dim: int = 2,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.channel = channel
        self.kernels = kernels
        self.T = T
        self.n_dim = n_dim

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the update of the channel by the kernels.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        distance = sqrt(sum(displacement(axis) ** 2 for axis in range(self.n_dim)))
        total = sum(kernel[2] for kernel in self.kernels)
        growth: Expression = 0
        for source, radius, weight, peaks, mu, sigma in self.kernels:
            kernel = self._shell(distance / radius, peaks) / self._norm(radius, peaks)
            potential = convolve(getattr(s.target, source), kernel, math.ceil(radius))
            growth += weight / total * (2 * exp(-((potential - mu) ** 2) / (2 * sigma**2)) - 1)
        value = getattr(s.target, self.channel)
        return [UpdateLaw(value, clip(value + growth / self.T, 0, 1))]

    @staticmethod
    def _shell(q: Expression, peaks: Sequence[float]) -> Expression:
        """Return the rings of a kernel at the distance `q` in units of its radius, zero outside the unit ball."""
        rings = len(peaks)
        shell: Expression = 0
        for index, peak in enumerate(peaks):
            x = rings * q - index
            inside = both(x > 0, x < 1)
            shell += where(inside, peak * exp(4 - 1 / where(inside, x * (1 - x), 1)), 0)
        return shell

    def _norm(self, radius: float, peaks: Sequence[float]) -> float:
        """Return the sum of the rings of a kernel over the sites within its radius."""
        steps = np.arange(-math.ceil(radius), math.ceil(radius) + 1)
        grid = np.meshgrid(*[steps] * self.n_dim, indexing='ij')
        q: Array = np.sqrt(sum(axis**2 for axis in grid)) / radius
        rings = len(peaks)
        values = np.zeros_like(q)
        for index, peak in enumerate(peaks):
            x = rings * q - index
            inside = (x > 0) & (x < 1)
            values += np.where(inside, peak * np.exp(4 - 1 / np.where(inside, x * (1 - x), 1)), 0)
        return float(values.sum())
