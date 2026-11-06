"""Test the fast spectral collision integral against direct quadrature."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from scipy.interpolate import RegularGridInterpolator

from skcomplex.core.engine.problems import CollisionKernel
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import RunStateSpec

type CrossSection = Callable[..., npt.NDArray[np.float64]]


def collision_by_quadrature(
    density: RegularGridInterpolator,
    velocity: npt.NDArray[np.float64],
    cross_section: CrossSection,
    extent: float,
) -> float:
    """Return the collision integral at one velocity by direct quadrature over relative velocities and angles."""
    nodes = np.linspace(-extent, extent, 81)
    step = nodes[1] - nodes[0]
    angles = 2 * np.pi * np.arange(48) / 48
    gx, gy = np.meshgrid(nodes, nodes, indexing='ij')
    speed = np.hypot(gx, gy)
    total = 0.0
    for angle in angles:
        sx, sy = np.cos(angle), np.sin(angle)
        cosine = np.where(speed > 0, (gx * sx + gy * sy) / np.where(speed > 0, speed, 1.0), 1.0)
        star = velocity - np.stack([gx, gy], axis=-1)
        after = velocity - (np.stack([gx, gy], axis=-1) - speed[..., None] * np.array([sx, sy])) / 2
        partner = velocity + star - after
        weight = cross_section(speed, cosine) * 2 * np.pi / angles.size * step**2
        total += float(np.sum(weight * (density(after) * density(partner) - density(velocity) * density(star))))
    return total


@pytest.mark.parametrize(
    'cross_section',
    [lambda _speed, _cosine: 1 / (2 * np.pi), lambda speed, _cosine: speed / (2 * np.pi)],
)
def test_run_agrees_with_direct_quadrature_on_a_coarse_grid(cross_section: CrossSection) -> None:
    """Agree with direct quadrature of the collision integral within 2 percent, for Maxwell molecules and hard disks."""
    size, half_width = 24, 6.0
    step = 2 * half_width / size
    centres = -half_width + (np.arange(size) + 0.5) * step
    vx, vy = np.meshgrid(centres, centres, indexing='ij')
    values = np.exp(-((vx - 1.0) ** 2) - vy**2) + np.exp(-((vx + 1.0) ** 2) - vy**2)
    kernel = CollisionKernel((size, size), (step, step), cross_section).prepare()
    rates = kernel.run(RunStateSpec(0.0, values.ravel(), np.zeros(0))).values_.reshape(size, size)
    density = RegularGridInterpolator((centres, centres), values, method='cubic', bounds_error=False, fill_value=0.0)
    probes = [(12, 12), (15, 12), (12, 15), (9, 13)]
    direct = np.array(
        [
            collision_by_quadrature(density, np.array([centres[i], centres[j]]), cross_section, 2 * half_width)
            for i, j in probes
        ],
    )
    computed = np.array([rates[i, j] for i, j in probes])
    assert np.abs(computed - direct).max() < 0.02 * np.abs(direct).max()


def test_check_rejects_values_it_cannot_take() -> None:
    """Raise a named error for a shape without two axes or a cross-section that is no function."""
    with pytest.raises(ValidationError, match='two velocity axes'):
        CollisionKernel((4,), (1.0,), lambda _speed, _cosine: 1.0).prepare()
    with pytest.raises(ValidationError, match='is a function'):
        CollisionKernel((4, 4), (1.0, 1.0), 1.0).prepare()
