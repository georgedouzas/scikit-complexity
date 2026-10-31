"""Billiard models: free particles reflecting off the walls of Sinai and mushroom tables."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, CrossingLaw, RateLaw
from ....core.laws.utils import where
from ....core.system import System
from ....errors import ValidationError
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model

SHAPES = ('sinai', 'mushroom')


def _sinai_walls(size: float, radius: float, **_: float) -> list[np.ndarray]:
    """Return the square wall and the central disc of a Sinai table as closed polylines."""
    square = size * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]])
    angles = np.linspace(0.0, 2 * np.pi, 100)
    return [square, radius * np.column_stack([np.cos(angles), np.sin(angles)])]


class SinaiBilliardElement(Element):
    """Free particle in a square table of half side `size` with a central disc of radius `radius`, the Sinai billiard.

    The particle flies straight and reflects specularly off the four walls and the disc, which makes its motion
    ergodic and chaotic.

    Args:
        label:
            The label of the particle.

        visual:
            How the particle is drawn.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        vx:
            The initial first velocity.

        vy:
            The initial second velocity.

        size:
            The half side of the square.

        radius:
            The radius of the disc.

    Attributes:
        default_visual_:
            The visual of the particle when it sets none.

    Examples:
        >>> from skcomplex.models.physics.nonlinear_oscillators import SinaiBilliardElement
        >>> SinaiBilliardElement('ball').radius
        0.4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec(
            'bodies', {'positions': ('x', 'y'), 'trail': None, 'outline': _sinai_walls},
        ),
        *,
        x: Annotated[float, StateSpec()] = 0.1,
        y: Annotated[float, StateSpec()] = 0.6,
        vx: Annotated[float, StateSpec()] = 0.6,
        vy: Annotated[float, StateSpec()] = 0.8,
        size: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        radius: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.4,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.size = size
        self.radius = radius

    def _reflect(self, namespace: SimpleNamespace) -> dict[Expression, Expression]:
        """Return the velocity after a specular reflection off the disc, whose normal is the position."""
        s = namespace
        projection = (s.vx * s.x + s.vy * s.y) / (s.x**2 + s.y**2)
        return {s.vx: s.vx - 2 * projection * s.x, s.vy: s.vy - 2 * projection * s.y}

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the free flight and the reflections off the walls and the disc.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rates of the position and the reflection events.
        """
        s = namespace
        return [
            RateLaw(s.x, s.vx),
            RateLaw(s.y, s.vy),
            CrossingLaw(s.x - s.size, {s.vx: -s.vx}, direction=1, label='right wall'),
            CrossingLaw(s.x + s.size, {s.vx: -s.vx}, direction=-1, label='left wall'),
            CrossingLaw(s.y - s.size, {s.vy: -s.vy}, direction=1, label='top wall'),
            CrossingLaw(s.y + s.size, {s.vy: -s.vy}, direction=-1, label='bottom wall'),
            CrossingLaw(s.x**2 + s.y**2 - s.radius**2, self._reflect(s), direction=-1, label='disc'),
        ]


class MushroomBilliardElement(Element):
    """Free particle in a mushroom table: a half disc cap of radius `radius` on a stem of half width `width`.

    The particle flies straight and reflects specularly off the cap, the floor of the cap beside the stem, the walls of
    the stem, and its floor at depth `height`. Its phase space mixes regular orbits that stay in the cap with chaotic
    orbits that enter the stem.

    Args:
        label:
            The label of the particle.

        visual:
            How the particle is drawn, or `None` for `default_visual_`.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        vx:
            The initial first velocity.

        vy:
            The initial second velocity.

        radius:
            The radius of the cap.

        width:
            The half width of the stem.

        height:
            The height of the stem.

    Attributes:
        default_visual_:
            The visual of the particle when it sets none.

    Examples:
        >>> from skcomplex.models.physics.nonlinear_oscillators import MushroomBilliardElement
        >>> MushroomBilliardElement('ball').width
        0.4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.1,
        y: Annotated[float, StateSpec()] = 0.6,
        vx: Annotated[float, StateSpec()] = 0.6,
        vy: Annotated[float, StateSpec()] = 0.8,
        radius: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        width: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.4,
        height: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.width = width
        self.height = height

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the free flight and the reflections off the parts of the lines of the table that are walls.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rates of the position and the reflection events.
        """
        s = namespace
        projection = (s.vx * s.x + s.vy * s.y) / (s.x**2 + s.y**2)
        in_cap, in_stem = s.y > 0, s.y < 0
        cap = {
            s.vx: where(in_cap, s.vx - 2 * projection * s.x, s.vx),
            s.vy: where(in_cap, s.vy - 2 * projection * s.y, s.vy),
        }
        return [
            RateLaw(s.x, s.vx),
            RateLaw(s.y, s.vy),
            CrossingLaw(s.x**2 + s.y**2 - s.radius**2, cap, direction=1, label='cap'),
            CrossingLaw(s.y, {s.vy: where(abs(s.x) > s.width, -s.vy, s.vy)}, direction=-1, label='cap floor'),
            CrossingLaw(s.x - s.width, {s.vx: where(in_stem, -s.vx, s.vx)}, direction=1, label='right stem wall'),
            CrossingLaw(s.x + s.width, {s.vx: where(in_stem, -s.vx, s.vx)}, direction=-1, label='left stem wall'),
            CrossingLaw(s.y + s.height, {s.vy: -s.vy}, direction=-1, label='stem floor'),
        ]


@register_model(
    field='physics/nonlinear_oscillators',
    tags=('events', 'billiard', 'chaos', 'mixed phase space'),
    references=('Sinai 1970', 'Bunimovich 2001', 'Datseris and Parlitz 2022'),
    defaults={'until': 20.0, 'dt': 0.05},
)
def billiards(
    shape: str = 'sinai',
    x: float = 0.1,
    y: float = 0.6,
    vx: float = 0.6,
    vy: float = 0.8,
    *,
    size: float = 1.0,
    disc_radius: float = 0.4,
    cap_radius: float = 1.0,
    width: float = 0.4,
    height: float = 1.0,
) -> System:
    """Return a particle moving freely on a billiard table and reflecting specularly off its walls.

    The Sinai table is the square `[-size, size]²` with a disc of radius `disc_radius` at its center, and its motion is
    chaotic. The mushroom table is the upper half of a disc of radius `cap_radius` on a stem of half width `width` and
    height `height`, and its phase space mixes regular orbits that stay in the cap with chaotic orbits that enter the
    stem. Every bounce is an event, and the speed is conserved.

    Args:
        shape:
            `'sinai'` or `'mushroom'`.

        x:
            The initial horizontal position.

        y:
            The initial vertical position.

        vx:
            The initial horizontal velocity.

        vy:
            The initial vertical velocity.

        size:
            The half side of the Sinai table.

        disc_radius:
            The radius of the disc of the Sinai table.

        cap_radius:
            The radius of the cap of the mushroom.

        width:
            The half width of the stem of the mushroom.

        height:
            The height of the stem of the mushroom.

    Returns:
        system:
            The model, with the position `particle.x`, `particle.y` and the velocity `particle.vx`, `particle.vy`.

    Raises:
        ValidationError:
            The shape is unknown.

    Examples:
        >>> from skcomplex.models.physics.nonlinear_oscillators import billiards
        >>> type(billiards('mushroom').elements[0]).__name__
        'MushroomBilliardElement'
    """
    if shape == 'sinai':
        table = SinaiBilliardElement('particle', x=x, y=y, vx=vx, vy=vy, size=size, radius=disc_radius)
        return System(elements=[table])
    if shape == 'mushroom':
        table = MushroomBilliardElement(
            'particle',
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            radius=cap_radius,
            width=width,
            height=height,
        )
        return System(elements=[table])
    error_msg = f'Unknown billiard shape {shape!r}. Use one of {", ".join(SHAPES)}.'
    raise ValidationError(error_msg)
