"""Define random affine maps, which move a point of the plane by one of several affine maps drawn at random."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression, vector
from sage.modules.free_module_element import FreeModuleElement

from ....core.interactions import MapInteraction
from ....core.laws.utils import categorical, where
from ....errors import ValidationError
from ....specs import RoleSpec
from .types import AffineMap

AFFINE_COEFFICIENTS = 6
SIERPINSKI_MAPS: tuple[AffineMap, ...] = (
    (0.5, 0.0, 0.0, 0.5, 0.0, 0.0),
    (0.5, 0.0, 0.0, 0.5, 0.5, 0.0),
    (0.5, 0.0, 0.0, 0.5, 0.25, 0.5),
)


class RandomAffineMapInteraction(MapInteraction):
    """Map of a point of the plane by one affine map of an iterated function system, drawn anew at every step.

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
        >>> from skcomplex.models.mathematics.fractals import RandomAffineMapInteraction
        >>> game = RandomAffineMapInteraction('game', target='points', maps=[(0.5, 0, 0, 0.5, 0, 0)], weights=[1.0])
        >>> game.check().maps
        [(0.5, 0, 0, 0.5, 0, 0)]
    """

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

    def _check(self) -> None:
        """Raise an error for weights that do not match the maps, or maps without six coefficients.

        Raises:
            ValidationError:
                The number of weights differs from the number of maps, or a map does not have six coefficients.
        """
        super()._check()
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
