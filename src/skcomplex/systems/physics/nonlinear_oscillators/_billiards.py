"""Billiard systems.

Free particles reflect off the walls of Sinai and mushroom tables and off three open discs.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.system import BaseSystem, System
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..common.mechanics import ReflectionInteraction


class MushroomBilliardElement(Element):
    """Free particle in a mushroom table.

    The table is a half disc cap of radius `radius` on a stem of half width `width`.

    The particle flies straight, and reflects specularly off the cap, the floor of the cap beside the stem, the walls of
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
        >>> from skcomplex.systems.physics.nonlinear_oscillators import MushroomBilliardElement
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
        """Return the free flight `x' = vx`, `y' = vy` of the particle.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rates of the position.
        """
        s = namespace
        return [RateLaw(s.x, s.vx), RateLaw(s.y, s.vy)]


def _sinai_walls(size: float, radius: float, **_: float) -> list[np.ndarray]:
    """Return the square wall and central disc of a Sinai table as closed polylines."""
    square = size * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]])
    angles = np.linspace(0.0, 2 * np.pi, 100)
    return [square, radius * np.column_stack([np.cos(angles), np.sin(angles)])]


class SinaiBilliardElement(Element):
    """Free particle in the Sinai billiard.

    The table is a square of half side `size` with a central disc of radius `radius`.

    The particle flies straight, and its specular reflections off the four walls and the disc make its motion ergodic
    and chaotic.

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
        >>> from skcomplex.systems.physics.nonlinear_oscillators import SinaiBilliardElement
        >>> SinaiBilliardElement('ball').radius
        0.4
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec(
            'bodies',
            {'positions': ('x', 'y'), 'trail': None, 'outline': _sinai_walls},
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

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the free flight `x' = vx`, `y' = vy` of the particle.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rates of the position.
        """
        s = namespace
        return [RateLaw(s.x, s.vx), RateLaw(s.y, s.vy)]


ANGLES = (-math.pi / 2, math.pi / 6, 5 * math.pi / 6)
"""The polar angles of the centers of the three discs, one below the origin and two above it."""


def _three_disk_walls(spacing: float, radius: float, **_: float) -> list[np.ndarray]:
    """Return the three discs of an open three-disk billiard as closed polylines."""
    angles = np.linspace(0.0, 2 * np.pi, 100)
    circle = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    distance = spacing / math.sqrt(3)
    return [circle + distance * np.array([math.cos(angle), math.sin(angle)]) for angle in ANGLES]


class ThreeDiskBilliardElement(Element):
    """Free particle scattered by three discs of radius `radius`.

    The discs sit at the corners of a triangle of side `spacing`.

    The centers of the discs lie at the distance `spacing / √3` from the origin, at the polar angles `-π / 2`,
    `π / 6`, and `5 π / 6`. The particle flies straight and reflects specularly off the discs. The billiard is open,
    so almost every particle leaves the scattering region after a finite number of collisions, and the number of
    collisions depends on the incoming direction in a fractal way.

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

        spacing:
            The distance between the centers of two discs.

        radius:
            The radius of the discs.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import ThreeDiskBilliardElement
        >>> ThreeDiskBilliardElement('ball').spacing
        2.4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec(
            'bodies',
            {'positions': ('x', 'y'), 'trail': None, 'outline': _three_disk_walls},
        ),
        *,
        x: Annotated[float, StateSpec()] = 0.1,
        y: Annotated[float, StateSpec()] = 0.6,
        vx: Annotated[float, StateSpec()] = 0.6,
        vy: Annotated[float, StateSpec()] = 0.8,
        spacing: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.4,
        radius: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.4,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.spacing = spacing
        self.radius = radius

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the free flight `x' = vx`, `y' = vy` of the particle.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rates of the position.
        """
        s = namespace
        return [RateLaw(s.x, s.vx), RateLaw(s.y, s.vy)]


SHAPES = ('sinai', 'mushroom', 'three_disk')


def _wall(
    label: str,
    guard: Callable[[SimpleNamespace], object],
    direction: int,
    region: Callable[[SimpleNamespace], object] | None = None,
) -> ReflectionInteraction:
    """Return the specular reflection of the particle off a wall of the table."""
    return ReflectionInteraction(
        label,
        target='particle',
        guard=guard,
        positions=('x', 'y'),
        velocities=('vx', 'vy'),
        direction=direction,
        region=region,
    )


class BilliardsSystem(BaseSystem):
    """A particle moving freely on a billiard table.

    The particle reflects specularly off the walls of the table.

    The Sinai table is the square `[-size, size]²` with a disc of radius `disc_radius` at its center, and its motion is
    chaotic. The mushroom table is the upper half of a disc of radius `cap_radius` on a stem of half width `width` and
    height `height`, and its phase space mixes regular orbits that stay in the cap with chaotic orbits that enter the
    stem. The three-disk table is open: three discs of radius `disc_radius` sit at the corners of an equilateral
    triangle of side `spacing` centered at the origin, with one disc below the origin. A particle that comes from far
    away is scattered chaotically between the discs and leaves again, and the number of its collisions and its outgoing
    angle depend on its incoming direction through a fractal scattering function. Every bounce is an event, and the
    speed is conserved. The system has the position `particle.x`, `particle.y` and the velocity `particle.vx`,
    `particle.vy`.

    Args:
        shape:
            `'sinai'`, `'mushroom'`, or `'three_disk'`.

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
            The radius of the disc of the Sinai table and of the discs of the three-disk table.

        cap_radius:
            The radius of the cap of the mushroom.

        width:
            The half width of the stem of the mushroom.

        height:
            The height of the stem of the mushroom.

        spacing:
            The distance between the centers of the discs of the three-disk table.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The shape is unknown.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import BilliardsSystem
        >>> type(BilliardsSystem('mushroom').prepare().system_.elements[0]).__name__
        'MushroomBilliardElement'
    """

    field = 'physics/nonlinear_oscillators'
    topics = ('events', 'billiard', 'chaos', 'mixed phase space')
    references = ('Sinai 1970', 'Bunimovich 2001', 'Gaspard and Rice 1989', 'Datseris and Parlitz 2022')
    until = 20.0
    dt = 0.05

    def __init__(
        self,
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
        spacing: float = 2.4,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.size = size
        self.disc_radius = disc_radius
        self.cap_radius = cap_radius
        self.width = width
        self.height = height
        self.spacing = spacing

    @staticmethod
    def _disc(label: str, angle: float) -> ReflectionInteraction:
        """Return the reflection off a disc of the three-disk table.

        The center of the disc lies at a polar angle.
        """
        cosine, sine = math.cos(angle), math.sin(angle)

        def guard(s: SimpleNamespace) -> object:
            distance = s.target.spacing / math.sqrt(3)
            return (s.target.x - distance * cosine) ** 2 + (s.target.y - distance * sine) ** 2 - s.target.radius**2

        return _wall(label, guard, -1)

    def _build_system(self) -> System:
        """Return a particle moving freely on a billiard table.

        The particle reflects specularly off the walls of the table.
        """
        if self.shape == 'sinai':
            table = SinaiBilliardElement(
                'particle',
                x=self.x,
                y=self.y,
                vx=self.vx,
                vy=self.vy,
                size=self.size,
                radius=self.disc_radius,
            )
            walls = [
                _wall('right_wall', lambda s: s.target.x - s.target.size, 1),
                _wall('left_wall', lambda s: s.target.x + s.target.size, -1),
                _wall('top_wall', lambda s: s.target.y - s.target.size, 1),
                _wall('bottom_wall', lambda s: s.target.y + s.target.size, -1),
                _wall('disc', lambda s: s.target.x**2 + s.target.y**2 - s.target.radius**2, -1),
            ]
            return System(self.label, elements=[table], interactions=walls)
        if self.shape == 'mushroom':
            table = MushroomBilliardElement(
                'particle',
                x=self.x,
                y=self.y,
                vx=self.vx,
                vy=self.vy,
                radius=self.cap_radius,
                width=self.width,
                height=self.height,
            )
            walls = [
                _wall('cap', lambda s: s.target.x**2 + s.target.y**2 - s.target.radius**2, 1, lambda s: s.target.y > 0),
                _wall('cap_floor', lambda s: s.target.y, -1, lambda s: abs(s.target.x) > s.target.width),
                _wall('right_stem_wall', lambda s: s.target.x - s.target.width, 1, lambda s: s.target.y < 0),
                _wall('left_stem_wall', lambda s: s.target.x + s.target.width, -1, lambda s: s.target.y < 0),
                _wall('stem_floor', lambda s: s.target.y + s.target.height, -1),
            ]
            return System(self.label, elements=[table], interactions=walls)
        if self.shape == 'three_disk':
            table = ThreeDiskBilliardElement(
                'particle',
                x=self.x,
                y=self.y,
                vx=self.vx,
                vy=self.vy,
                spacing=self.spacing,
                radius=self.disc_radius,
            )
            walls = [self._disc(f'disc_{index}', angle) for index, angle in enumerate(ANGLES)]
            return System(self.label, elements=[table], interactions=walls)
        error_msg = f'Unknown billiard shape {self.shape!r}. Use one of {", ".join(SHAPES)}.'
        raise ValidationError(error_msg)
