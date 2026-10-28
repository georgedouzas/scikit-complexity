"""Fourth-order exponential time differencing Runge-Kutta of Cox and Matthews, for spectral fields."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ...core.errors import SolverError
from ...core.types import Vector
from ._split import _SplitSolver
from ._split_operator import Complex

_CONTOUR_POINTS = 64


class ETDRK4Solver(_SplitSolver):
    """Fourth-order exponential time differencing Runge-Kutta of Cox and Matthews, for spectral fields.

    The linear part is integrated exactly per Fourier mode and the remainder by a fourth-order scheme, with the
    coefficients computed by contour integrals as Kassam and Trefethen propose. Coupled states of a field use a small
    matrix exponential per mode. States outside spectral fields follow classical RK4.

    Args:
        dt:
            The fixed internal step.

    Examples:
        >>> from skcomplex.engine.solvers import ETDRK4Solver
        >>> ETDRK4Solver(dt=0.25).dt
        0.25
    """

    def _built(self, p: Vector) -> None:
        """Check that the linear part is spectral and forget its coefficients."""
        if self.operator_.sparse is not None:
            error_msg = 'ETDRK4Solver needs the linear part on spectral fields. Use IMEX2Solver or CrankNicolsonSolver.'
            raise SolverError(error_msg)
        self._coefficients: dict[float, list[list[Complex]]] = {}

    def _matrices(self, h: float) -> list[list[Complex]]:
        """Return the six ETDRK4 matrices of every block for a step."""
        if h not in self._coefficients:
            blocks = []
            for block in self.operator_.blocks:
                values, vectors = np.linalg.eig(h * block.matrix)
                inverse = np.linalg.inv(vectors)
                blocks.append(
                    [np.einsum('...ij,...j,...jk->...ik', vectors, item, inverse) for item in self._phi(values, h)],
                )
            self._coefficients[h] = blocks
        return self._coefficients[h]

    def _map(self, index: int, h: float, x: Vector) -> Vector:
        """Return one of the six ETDRK4 maps applied to a state vector, with its L = 0 limit outside blocks."""
        scalars = [1.0, 1.0, h / 2, h / 6, h / 6, h / 6]
        out = scalars[index] * x
        for block, matrices in zip(self.operator_.blocks, self._matrices(h), strict=True):
            block.apply(matrices[index], x, out)
        return out

    def _step(self, t: float, x: Vector, p: Vector, h: float) -> Vector:
        """Return the state after one ETDRK4 step."""
        n_x = self._remainder(t, x, p)
        half = self._map(1, h, x)
        a = half + self._map(2, h, n_x)
        n_a = self._remainder(t + h / 2, a, p)
        b = half + self._map(2, h, n_a)
        n_b = self._remainder(t + h / 2, b, p)
        c = self._map(1, h, a) + self._map(2, h, 2 * n_b - n_x)
        n_c = self._remainder(t + h, c, p)
        return self._map(0, h, x) + self._map(3, h, n_x) + 2 * self._map(4, h, n_a + n_b) + self._map(5, h, n_c)

    @staticmethod
    def _phi(z: Complex, h: float) -> list[Complex]:
        """Return `e^z`, `e^(z/2)`, and the coefficients `Q`, `f1`, `f2`, `f3` of ETDRK4 by contour integrals.

        The integrals average over a full circle of radius 1 around every eigenvalue, which avoids the cancellation of
        the formulas near zero for complex eigenvalues as well as real ones.
        """
        roots = np.exp(2j * np.pi * (np.arange(1, _CONTOUR_POINTS + 1) - 0.5) / _CONTOUR_POINTS)
        points = z[..., None] + roots
        exponential = np.exp(points)
        q = h * np.mean((np.exp(points / 2) - 1) / points, axis=-1)
        f1 = h * np.mean((-4 - points + exponential * (4 - 3 * points + points**2)) / points**3, axis=-1)
        f2 = h * np.mean((2 + points + exponential * (-2 + points)) / points**3, axis=-1)
        f3 = h * np.mean((-4 - 3 * points - points**2 + exponential * (4 - points)) / points**3, axis=-1)
        return [np.exp(z), np.exp(z / 2), q, f1, f2, f3]
