"""Billiard systems: free particles reflecting off the walls of Sinai and mushroom tables."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace

from ....core.components.system import System
from ....core.errors import ValidationError
from ..._catalog import register_system
from ..common.mechanics import ReflectionInteraction
from ._mushroom_billiard import MushroomBilliardElement
from ._sinai_billiard import SinaiBilliardElement

SHAPES = ('sinai', 'mushroom')


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


@register_system(
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
            The system, with the position `particle.x`, `particle.y` and the velocity `particle.vx`, `particle.vy`.

    Raises:
        ValidationError:
            The shape is unknown.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import billiards
        >>> type(billiards('mushroom').elements[0]).__name__
        'MushroomBilliardElement'
    """
    if shape == 'sinai':
        table = SinaiBilliardElement('particle', x=x, y=y, vx=vx, vy=vy, size=size, radius=disc_radius)
        walls = [
            _wall('right_wall', lambda s: s.target.x - s.target.size, 1),
            _wall('left_wall', lambda s: s.target.x + s.target.size, -1),
            _wall('top_wall', lambda s: s.target.y - s.target.size, 1),
            _wall('bottom_wall', lambda s: s.target.y + s.target.size, -1),
            _wall('disc', lambda s: s.target.x**2 + s.target.y**2 - s.target.radius**2, -1),
        ]
        return System(elements=[table], interactions=walls)
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
        walls = [
            _wall('cap', lambda s: s.target.x**2 + s.target.y**2 - s.target.radius**2, 1, lambda s: s.target.y > 0),
            _wall('cap_floor', lambda s: s.target.y, -1, lambda s: abs(s.target.x) > s.target.width),
            _wall('right_stem_wall', lambda s: s.target.x - s.target.width, 1, lambda s: s.target.y < 0),
            _wall('left_stem_wall', lambda s: s.target.x + s.target.width, -1, lambda s: s.target.y < 0),
            _wall('stem_floor', lambda s: s.target.y + s.target.height, -1),
        ]
        return System(elements=[table], interactions=walls)
    error_msg = f'Unknown billiard shape {shape!r}. Use one of {", ".join(SHAPES)}.'
    raise ValidationError(error_msg)
