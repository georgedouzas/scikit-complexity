"""Fractal systems of iterated function systems drawn by the chaos game."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, vector
from sage.modules.free_module_element import FreeModuleElement

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MapInteraction
from ....core.components.laws.utils import categorical, where
from ....core.components.spaces import BaseSpace, ContinuousSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ValidationError
from ....core.specs import RoleSpec, StateSpec, VisualSpec
from .types import AffineMap

AFFINE_COEFFICIENTS = 6
SIERPINSKI_MAPS: tuple[AffineMap, ...] = (
    (0.5, 0.0, 0.0, 0.5, 0.0, 0.0),
    (0.5, 0.0, 0.0, 0.5, 0.5, 0.0),
    (0.5, 0.0, 0.0, 0.5, 0.25, 0.5),
)
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


class RandomAffineMapInteraction(MapInteraction):
    """Map of a point of the plane by one affine map of an iterated function system.

    The map is drawn anew at every step.

    Map `k` takes `(x, y)` to `(a x + b y + e, c x + d y + f)` for its coefficients `(a, b, c, d, e, f)`, and is drawn
    with a probability proportional to its weight, independently for every member of a bound group.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose position the map sets.

        state:
            The name of the vector state of the target that holds the position.

        maps:
            The coefficients `(a, b, c, d, e, f)` of every affine map.

        weights:
            The weight of every map.

    Raises:
        ValidationError:
            The number of weights differs from the number of maps, or a map does not have six coefficients.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import RandomAffineMapInteraction
        >>> game = RandomAffineMapInteraction('game', target='points', maps=[(0.5, 0, 0, 0.5, 0, 0)], weights=[1.0])
        >>> game.maps
        [(0.5, 0, 0, 0.5, 0, 0)]
    """

    draws = True

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'p',
        maps: Sequence[AffineMap] = SIERPINSKI_MAPS,
        weights: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__(label, target=target, state=state, function=self._get_image)
        self.maps = maps
        self.weights = weights

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for invalid weights or maps.

        The weights must match the maps, and every map must have six coefficients.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The number of weights differs from the number of maps, or a map does not have six coefficients.
        """
        super()._prepare(context)
        if len(self.maps) != len(self.weights) or any(len(item) != AFFINE_COEFFICIENTS for item in self.maps):
            error_msg = (
                f'The random affine map {self.label!r} takes one weight per map and six coefficients per map, but got '
                f'{len(self.maps)} maps and {len(self.weights)} weights.'
            )
            raise ValidationError(error_msg)

    def _get_image(self, namespace: SimpleNamespace) -> FreeModuleElement:
        """Return the image of the position by the map drawn among the maps."""
        chosen = categorical('map', list(self.weights))
        position = getattr(namespace.target, self.state)
        x, y = position[0], position[1]
        images: list[tuple[Expression, Expression]] = [
            (a * x + b * y + e, c * x + d * y + f) for a, b, c, d, e, f in self.maps
        ]
        new_x, new_y = SR(images[-1][0]), SR(images[-1][1])
        for index in range(len(images) - 2, -1, -1):
            new_x = where(chosen == index, images[index][0], new_x)
            new_y = where(chosen == index, images[index][1], new_y)
        return vector(SR, [new_x, new_y])


class ChaosGameSystem(BaseSystem):
    """Points that draw the attractor of an iterated function system by the chaos game.

    At every step each point moves by one of the affine maps of the system, drawn with probabilities proportional to the
    weights. The maps contract, so after a few steps the cloud of points covers the attractor, the unique compact set
    that the union of the maps leaves unchanged. The Sierpiński triangle of three maps of ratio 1/2 has the box-counting
    dimension `log 3 / log 2`, and the Barnsley fern is the attractor of four maps. The system has the positions
    `points.p`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The preset is unknown.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import ChaosGameSystem
        >>> len(ChaosGameSystem('barnsley', n_points=10).prepare().system_.interactions[0].maps)
        4
    """

    field = 'mathematics/fractals'
    topics = ('fractal', 'iterated function system', 'chaos game', 'discrete map')
    references = ('Barnsley 1988', 'Dilão 2023')
    until = 30
    dt = 1

    def __init__(
        self,
        preset: str = 'sierpinski',
        *,
        maps: Sequence[AffineMap] | None = None,
        weights: Sequence[float] | None = None,
        n_points: int = 20000,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.preset = preset
        self.maps = maps
        self.weights = weights
        self.n_points = n_points
        self.random_state = random_state

    @staticmethod
    def _get_system_of(
        label: str,
        maps: Sequence[AffineMap],
        weights: Sequence[float],
        n_points: int,
        random_state: int | np.random.Generator | None,
    ) -> CompositeSystem:
        """Return a group of points started uniformly in the unit square.

        The points iterate an iterated function system.
        """
        starts = np.random.default_rng(random_state).random((n_points, 2))
        points = GroupElement('points', element=ChaosPointElement, size=n_points, values={'p': starts})
        game = RandomAffineMapInteraction(
            'game',
            target='points',
            maps=tuple(map(tuple, maps)),
            weights=tuple(weights),
        )
        return CompositeSystem(label, elements=[points], interactions=[game], space=ContinuousSpace(n_dim=2))

    def _build_system(self) -> CompositeSystem:
        """Return points that draw the attractor of an iterated function system.

        The points draw it by the chaos game.
        """
        if self.maps is None and self.preset not in PRESETS:
            error_msg = f'Unknown chaos game preset {self.preset!r}. Use one of {", ".join(PRESETS)}, or give the maps.'
            raise ValidationError(error_msg)
        chosen_maps, preset_weights = (
            PRESETS[self.preset] if self.maps is None else (self.maps, (1.0,) * len(self.maps))
        )
        return self._get_system_of(
            self.label,
            chosen_maps,
            self.weights if self.weights is not None else preset_weights,
            self.n_points,
            self.random_state,
        )
