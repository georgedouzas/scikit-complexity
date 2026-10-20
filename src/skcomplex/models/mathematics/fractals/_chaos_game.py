"""Fractal models: iterated function systems drawn by the chaos game."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
from sage.all__sagemath_symbolics import SR

from ....core.base import Expression, ValidationError, VisualSpec, state, vector
from ....core.model import Element, EuclideanSpace, Law, Symbols, System, Update, categorical, where
from ....core.runtime import register_model

type AffineMap = Sequence[float]

PRESETS: dict[str, tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]] = {
    'sierpinski': (
        ((0.5, 0.0, 0.0, 0.5, 0.0, 0.0), (0.5, 0.0, 0.0, 0.5, 0.5, 0.0), (0.5, 0.0, 0.0, 0.5, 0.25, 0.5)),
        (1.0, 1.0, 1.0),
    ),
    'barnsley': (
        (
            (0.0, 0.0, 0.0, 0.16, 0.0, 0.0),
            (0.85, 0.04, -0.04, 0.85, 0.0, 1.6),
            (0.2, -0.26, 0.23, 0.22, 0.0, 1.6),
            (-0.15, 0.28, 0.26, 0.24, 0.0, 0.44),
        ),
        (0.01, 0.85, 0.07, 0.07),
    ),
}


class ChaosPoint(Element):
    """Point of the plane moved at every step by one affine map of an iterated function system, drawn at random.

    Map `k` takes `(x, y)` to `(a x + b y + e, c x + d y + f)` for its coefficients `(a, b, c, d, e, f)`, and is drawn
    with a probability proportional to its weight.

    Args:
        label:
            The label of the point or the group.

        p:
            The initial position.

        maps:
            The coefficients `(a, b, c, d, e, f)` of every affine map.

        weights:
            The weight of every map.

    Examples:
        >>> from skcomplex.models.mathematics.fractals import ChaosPoint
        >>> len(ChaosPoint('point').maps)
        3
    """

    visual: ClassVar[VisualSpec] = VisualSpec('points', {'position': 'p', 'radius': 0.004, 'states': ()})
    p: object = state(default=0.0, vector=True)
    maps: Sequence[AffineMap] = PRESETS['sierpinski'][0]
    weights: Sequence[float] = PRESETS['sierpinski'][1]

    def laws(self, s: Symbols) -> list[Law]:
        """Return the move by the drawn map.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        chosen = categorical('map', list(self.weights))
        x, y = s.p[0], s.p[1]
        images: list[tuple[Expression, Expression]] = [
            (a * x + b * y + e, c * x + d * y + f) for a, b, c, d, e, f in self.maps
        ]
        new_x, new_y = SR(images[-1][0]), SR(images[-1][1])
        for index in range(len(images) - 2, -1, -1):
            new_x = where(chosen == index, images[index][0], new_x)
            new_y = where(chosen == index, images[index][1], new_y)
        return [Update(s.p, vector([new_x, new_y]))]


def _system_of(maps: Sequence[AffineMap], weights: Sequence[float], n_points: int, random_state: object) -> System:
    """Return a group of points started uniformly in the unit square, which iterate an iterated function system."""
    if len(maps) != len(weights) or any(len(item) != 6 for item in maps):  # noqa: PLR2004
        error_msg = (
            f'The chaos game takes one weight per map and six coefficients per map, but got {len(maps)} maps and '
            f'{len(weights)} weights.'
        )
        raise ValidationError(error_msg)
    starts = np.random.default_rng(random_state).random((n_points, 2))  # type: ignore[arg-type]
    points = ChaosPoint.group('points', n_points, p=starts, maps=tuple(map(tuple, maps)), weights=tuple(weights))
    return System(elements=[points], space=EuclideanSpace(2))


@register_model(
    field='mathematics/fractals',
    tags=('fractal', 'iterated function system', 'chaos game', 'discrete map'),
    references=('Barnsley 1988', 'Dilão 2023'),
    defaults={'until': 30, 'dt': 1},
)
def chaos_game(
    preset: str = 'sierpinski',
    *,
    maps: Sequence[AffineMap] | None = None,
    weights: Sequence[float] | None = None,
    n_points: int = 20000,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return points that draw the attractor of an iterated function system by the chaos game.

    At every step each point moves by one of the affine maps of the system, drawn with probabilities proportional to
    the weights. The maps contract, so after a few steps the cloud of points covers the attractor, the unique compact
    set that the union of the maps leaves unchanged. The Sierpiński triangle of three maps of ratio 1/2 has the
    box-counting dimension `log 3 / log 2`, and the Barnsley fern is the attractor of four maps.

    Args:
        preset:
            `'sierpinski'` or `'barnsley'`, used when `maps` is `None`.

        maps:
            The coefficients `(a, b, c, d, e, f)` of every map `(x, y) -> (a x + b y + e, c x + d y + f)`, or `None` for
            the preset.

        weights:
            The weight of every map, or `None` for the weights of the preset, or equal weights for given maps.

        n_points:
            The number of points, started uniformly in the unit square.

        random_state:
            The seed or generator of the initial points.

    Returns:
        system:
            The model, with the positions `points.p`.

    Raises:
        ValidationError:
            The preset is unknown, or the maps and weights do not match.

    Examples:
        >>> from skcomplex.models.mathematics.fractals import chaos_game
        >>> len(chaos_game('barnsley', n_points=10).elements[0].values['maps'])
        4
    """
    if maps is None and preset not in PRESETS:
        error_msg = f'Unknown chaos game preset {preset!r}. Use one of {", ".join(PRESETS)}, or give the maps.'
        raise ValidationError(error_msg)
    chosen_maps, preset_weights = PRESETS[preset] if maps is None else (maps, (1.0,) * len(maps))
    return _system_of(chosen_maps, weights if weights is not None else preset_weights, n_points, random_state)
