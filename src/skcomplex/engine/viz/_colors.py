"""Map values to colors on a sequential, a cyclic, or a categorical palette."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt

SERIES_COLORS = (
    (230, 159, 0),
    (86, 180, 233),
    (0, 158, 115),
    (240, 228, 66),
    (0, 114, 178),
    (213, 94, 0),
    (204, 121, 167),
    (170, 170, 170),
)
POINT_FLOOR = 0.25
COLOR_SCALE = np.array(
    [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
    dtype=float,
)
PALETTE = (
    np.stack(
        [
            np.interp(np.linspace(0.0, 1.0, 256), np.linspace(0.0, 1.0, len(COLOR_SCALE)), COLOR_SCALE[:, channel])
            for channel in range(3)
        ],
        axis=-1,
    )
    .round()
    .astype(np.uint8)
)
PHASE_SCALE = np.array(
    [[230, 90, 80], [225, 190, 70], [90, 190, 110], [70, 160, 210], [150, 100, 200], [230, 90, 80]],
    dtype=float,
)
PHASE_PALETTE = (
    np.stack(
        [
            np.interp(np.linspace(0.0, 1.0, 256), np.linspace(0.0, 1.0, len(PHASE_SCALE)), PHASE_SCALE[:, channel])
            for channel in range(3)
        ],
        axis=-1,
    )
    .round()
    .astype(np.uint8)
)


def _widened(span: tuple[float, float] | None, values: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Return a color range widened to cover values, starting from their extremes when there is no range yet."""
    low, high = float(np.min(values)), float(np.max(values))
    return (low, high) if span is None else (min(span[0], low), max(span[1], high))


def _phase_colors(phases: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    """Return the colors of phases in `(-pi, pi]` from a cyclic palette, so `-pi` and `pi` share one color."""
    levels = np.mod(phases + np.pi, 2 * np.pi) * (255.0 / (2 * np.pi))
    return PHASE_PALETTE[np.clip(levels, 0.0, 255.0).astype(np.uint8)]


def _levels_of(values: npt.NDArray[np.float64], low: float, high: float) -> npt.NDArray[np.uint8]:
    """Return the values as levels from 1 to 255 of a color map, with 0 left for no data."""
    if high <= low:
        return np.ones(values.shape, dtype=np.uint8)
    return (1 + np.clip((values - low) * (254.0 / (high - low)), 0.0, 254.0)).astype(np.uint8)


def palette_colors(values: npt.NDArray[np.float64], low: float, high: float) -> npt.NDArray[np.uint8]:
    """Return the colors of an array of values from a palette of 256 levels, fast enough for large images.

    Args:
        values:
            The values.

        low:
            The value of the darkest color.

        high:
            The value of the brightest color.

    Returns:
        colors:
            One RGB color per value, the darkest one for every value when the range is empty.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.viz import palette_colors
        >>> palette_colors(np.array([0.0, 1.0]), 0.0, 1.0).shape
        (2, 3)
    """
    if high <= low:
        return np.broadcast_to(PALETTE[0], (*values.shape, 3))
    levels = np.clip((values - low) * (255.0 / (high - low)), 0.0, 255.0).astype(np.uint8)
    return PALETTE[levels]


def colors_of(values: npt.NDArray[np.float64], low: float, high: float) -> npt.NDArray[np.uint8]:
    """Return one color per value on a scale from blue at `low` to yellow at `high`, bright enough to see on black.

    Args:
        values:
            The values.

        low:
            The value of the darkest color.

        high:
            The value of the brightest color.

    Returns:
        colors:
            One RGB color per value.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.viz import colors_of
        >>> colors_of(np.array([0.0, 1.0]), 0.0, 1.0).shape
        (2, 3)
    """
    fractions = np.clip((values - low) / (high - low), 0.0, 1.0) if high > low else np.zeros_like(values)
    fractions = POINT_FLOOR + (1.0 - POINT_FLOOR) * fractions
    anchors = np.linspace(0.0, 1.0, len(COLOR_SCALE))
    channels = [np.interp(fractions, anchors, COLOR_SCALE[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=-1).round().astype(np.uint8)
