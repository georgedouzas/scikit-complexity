"""Compile the binary collision integral of a density over two velocity axes by the fast spectral method."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable

import numpy as np
import scipy.fft

from ...errors import ValidationError
from ...specs import RunStateSpec
from ...types import Array
from ._base import BaseKernel

PADDING = (3 + math.sqrt(2)) / 2
RANK_TOLERANCE = 1e-10
PLANE = 2


class CollisionKernel(BaseKernel):
    """Binary collision integral of a density over two uniform velocity axes, by the method of Mouhot and Pareschi.

    The integral is `Q(f, f)(v) = int int B(|g|, cos t) (f(v') f(v'*) - f(v) f(v*)) ds dv*` in the Carleman form, an
    integral over the two orthogonal displacements `x = v' - v` and `y = v'* - v`, each up to twice the half-width of
    the grid. The weight of the pair `(|x|, |y|)` is split into terms of `|x|` times terms of `|y|` by a singular value
    decomposition, one term for Maxwell molecules, and the directions take a trapezoidal rule, so the integral is a set
    of FFT products on a grid padded by `(3 + sqrt(2)) / 2`. The mean rate is removed at every position, which conserves
    the density exactly.

    Args:
        dimensions:
            The number of points of every position axis and of the two velocity axes, the velocity axes last.

        spacings:
            The widths of the cells of the two velocity axes.

        cross_section:
            The function `B(speed, cosine)` of arrays of the relative speed and the cosine of the deflection angle.

        angles:
            The number of directions of the displacements, or `None` for one per two velocity points of an axis, at
            least 16.

    Attributes:
        padded_:
            The shape of the padded velocity grid.

        gains_:
            The Fourier weights of `x` and of `y` of every direction and term, stacked over the directions and terms.

        loss_:
            The Fourier weight of the loss term.

        values_:
            The collision integral of the last run at every site.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.problems import CollisionKernel
        >>> from skcomplex.core.specs import RunStateSpec
        >>> kernel = CollisionKernel((8, 8), (1.0, 1.0), lambda speed, cosine: 1 / (2 * np.pi)).prepare()
        >>> f = np.exp(-(np.indices((8, 8)) - 3.5) ** 2).prod(axis=0)
        >>> rates = kernel.run(RunStateSpec(0.0, f.ravel(), np.zeros(0))).values_
        >>> abs(float(rates.sum())) < 1e-12
        True
    """

    gains_: tuple[Array, Array]
    loss_: Array
    padded_: tuple[int, int]
    values_: Array

    def __init__(
        self,
        dimensions: tuple[int, ...],
        spacings: tuple[float, float],
        cross_section: Callable[[Array, Array], object],
        angles: int | None = None,
    ) -> None:
        self.dimensions = dimensions
        self.spacings = spacings
        self.cross_section = cross_section
        self.angles = angles

    def _check(self) -> None:
        """Raise an error for dimensions without two velocity axes or a cross-section that is no function."""
        if len(self.dimensions) < PLANE or len(self.spacings) != PLANE:
            error_msg = f'A collision integral reads two velocity axes, but the shape is {self.dimensions}.'
            raise ValidationError(error_msg)
        if not callable(self.cross_section):
            error_msg = f'The cross-section of a collision integral is a function, not {self.cross_section!r}.'
            raise ValidationError(error_msg)

    def _split_weight(self, nodes: Array) -> tuple[Array, Array, Array]:
        """Return the terms of `|x|` and of `|y|` and the singular values of the weight of the Carleman form."""
        first, second = np.meshgrid(nodes, nodes, indexing='ij')
        squared = first**2 + second**2
        values = self.cross_section(np.sqrt(squared), (second**2 - first**2) / squared)
        weight = 2.0 * np.broadcast_to(np.asarray(values, dtype=float), squared.shape)
        left, singular, right = np.linalg.svd(weight)
        rank = max(int(np.sum(singular > RANK_TOLERANCE * singular[0])), 1)
        return left[:, :rank], right[:rank], singular[:rank]

    def _prepare(self, context: object | None = None) -> None:
        """Assemble the Fourier weights of the gain and the loss terms on the padded grid."""
        sizes = self.dimensions[-PLANE:]
        first_size, second_size = (scipy.fft.next_fast_len(math.ceil(size * PADDING), real=True) for size in sizes)
        self.padded_ = (first_size, second_size)
        reach = max(size * step for size, step in zip(sizes, self.spacings, strict=True))
        count = self.angles if self.angles is not None else max(max(sizes) // 2, 16)
        points, weights = np.polynomial.legendre.leggauss(2 * max(self.padded_))
        nodes, weights = reach * (points + 1) / 2, reach * weights / 2
        along_x, along_y, singular = self._split_weight(nodes)
        k1 = 2 * np.pi * np.fft.fftfreq(first_size, d=self.spacings[0])
        k2 = 2 * np.pi * np.fft.rfftfreq(second_size, d=self.spacings[1])
        first, second = np.meshgrid(k1, k2, indexing='ij')
        gain_x, gain_y = [], []
        for angle in np.pi * np.arange(count) / count:
            parallel = np.cos(np.multiply.outer(nodes, first * np.cos(angle) + second * np.sin(angle)))
            normal = np.cos(np.multiply.outer(nodes, second * np.cos(angle) - first * np.sin(angle)))
            gain_x.append(2 * np.tensordot(along_x * weights[:, None], parallel, axes=(0, 0)))
            gain_y.append(2 * np.tensordot(along_y * weights, normal, axes=(1, 0)))
        scale = (np.pi / count) * singular
        terms_x, terms_y = np.concatenate(gain_x), np.concatenate(gain_y)
        repeated = np.tile(scale, count)[:, None, None]
        self.gains_ = (terms_x * repeated, terms_y)
        self.loss_ = np.sum(terms_x * terms_y * repeated, axis=0)

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Set the collision integral at every site from the density at every site, the `x` of `inputs`."""
        state = self._state_of(inputs)
        sizes = self.dimensions[-PLANE:]
        grid = np.reshape(state.x, self.dimensions)
        axes = (-2, -1)
        spectrum = scipy.fft.rfft2(grid, s=self.padded_, axes=axes)
        window = (..., slice(sizes[0]), slice(sizes[1]))
        terms_x, terms_y = self.gains_
        gain = np.zeros(grid.shape)
        for first, second in zip(terms_x, terms_y, strict=True):
            left = scipy.fft.irfft2(first * spectrum, s=self.padded_, axes=axes)[window]
            right = scipy.fft.irfft2(second * spectrum, s=self.padded_, axes=axes)[window]
            gain += left * right
        loss = grid * scipy.fft.irfft2(self.loss_ * spectrum, s=self.padded_, axes=axes)[window]
        rates = gain - loss
        self.values_ = (rates - rates.mean(axis=axes, keepdims=True)).ravel()
