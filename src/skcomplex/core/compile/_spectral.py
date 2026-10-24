"""Return the Fourier multipliers of differential operators on periodic lattices."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools

import numpy as np
import numpy.typing as npt


@functools.cache
def _wavenumbers(dimensions: tuple[int, ...], spacings: tuple[float, ...]) -> tuple[npt.NDArray[np.float64], ...]:
    """Return the angular wavenumbers of the real Fourier modes of a periodic lattice, broadcast along every axis."""
    axes = [2 * np.pi * np.fft.fftfreq(size, d=step) for size, step in zip(dimensions[:-1], spacings, strict=False)]
    axes.append(2 * np.pi * np.fft.rfftfreq(dimensions[-1], d=spacings[-1]))
    return tuple(np.meshgrid(*axes, indexing='ij'))


@functools.cache
def _dealiasing_mask(dimensions: tuple[int, ...]) -> npt.NDArray[np.bool_]:
    """Return the Fourier modes the two-thirds rule keeps, those below two thirds of the largest along every axis."""
    axes = [np.abs(np.fft.fftfreq(size, d=1.0 / size)) < size / 3 for size in dimensions[:-1]]
    axes.append(np.fft.rfftfreq(dimensions[-1], d=1.0 / dimensions[-1]) < dimensions[-1] / 3)
    grids = np.meshgrid(*axes, indexing='ij')
    return np.logical_and.reduce(grids)


def spectral_multiplier(
    name: str,
    axis: int,
    dimensions: tuple[int, ...],
    spacings: tuple[float, ...],
) -> npt.NDArray[np.complex128]:
    """Return the Fourier multiplier of an operator on a periodic lattice.

    The Laplacian and its inverse act on every mode, so the Laplacian damps the Nyquist mode by `-|k|^2` like any other.
    A first derivative is zero at the Nyquist mode of its axis, and keeps only the modes of the two-thirds rule, which
    dealiases the derivatives of products.

    Args:
        name:
            `'laplacian'`, `'partial'`, or `'inverse_laplacian'`.

        axis:
            The axis of a derivative.

        dimensions:
            The lattice shape.

        spacings:
            The distance between sites along every axis.

    Returns:
        multiplier:
            The factor of every real Fourier mode.
    """
    waves = _wavenumbers(dimensions, spacings)
    squared = np.sum([wave**2 for wave in waves], axis=0)
    if name == 'laplacian':
        return -squared.astype(complex)
    if name == 'inverse_laplacian':
        with np.errstate(divide='ignore'):
            return np.where(squared > 0, -1.0 / np.where(squared > 0, squared, 1.0), 0.0).astype(complex)
    factor = 1j * waves[axis]
    if dimensions[axis] % 2 == 0:
        nyquist = np.isclose(np.abs(waves[axis]), np.pi / spacings[axis])
        factor = np.where(nyquist, 0.0, factor)
    return np.where(_dealiasing_mask(dimensions), factor, 0.0)
