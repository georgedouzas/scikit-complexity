"""Fractal systems: iterated function systems drawn by the chaos game."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

import numpy as np

from ....core.components.elements import Element, GroupElement
from ....core.components.spaces import ContinuousSpace
from ....core.components.system import System
from ....core.errors import ValidationError
from ....core.specs import StateSpec, VisualSpec
from ..._catalog import register_system
from ._random_affine_map import SIERPINSKI_MAPS, RandomAffineMapInteraction
from .types import AffineMap

PRESETS: dict[str, tuple[tuple[AffineMap, ...], tuple[float, ...]]] = {
    'sierpinski': (
        SIERPINSKI_MAPS,
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


class ChaosPointElement(Element):
    """Point of the plane whose position a random affine map moves at every step.

    Args:
        label:
            The label of the point or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        p:
            The initial position.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import ChaosPointElement
        >>> ChaosPointElement('point', p=(0.5, 0.5)).p
        (0.5, 0.5)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('points', {'position': 'p', 'radius': 0.004, 'states': ()}),
        *,
        p: Annotated[object, StateSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.p = p


def _system_of(
    maps: Sequence[AffineMap],
    weights: Sequence[float],
    n_points: int,
    random_state: int | np.random.Generator | None,
) -> System:
    """Return a group of points started uniformly in the unit square, which iterate an iterated function system."""
    starts = np.random.default_rng(random_state).random((n_points, 2))
    points = GroupElement('points', element=ChaosPointElement, size=n_points, values={'p': starts})
    game = RandomAffineMapInteraction(
        'game',
        target='points',
        maps=tuple(map(tuple, maps)),
        weights=tuple(weights),
    )
    return System(elements=[points], interactions=[game], space=ContinuousSpace(n_dim=2))


@register_system(
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
            The system, with the positions `points.p`.

    Raises:
        ValidationError:
            The preset is unknown.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import chaos_game
        >>> len(chaos_game('barnsley', n_points=10).interactions[0].maps)
        4
    """
    if maps is None and preset not in PRESETS:
        error_msg = f'Unknown chaos game preset {preset!r}. Use one of {", ".join(PRESETS)}, or give the maps.'
        raise ValidationError(error_msg)
    chosen_maps, preset_weights = PRESETS[preset] if maps is None else (maps, (1.0,) * len(maps))
    return _system_of(chosen_maps, weights if weights is not None else preset_weights, n_points, random_state)
