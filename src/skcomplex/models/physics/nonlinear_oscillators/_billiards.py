"""Billiard models: free particles reflecting off the walls of Sinai and mushroom tables."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.base import Domain, Expression, ValidationError, VisualSpec, parameter
from ....core.model import Element, Event, Law, Rate, Symbols, System, where
from ....core.runtime import register_model

SHAPES = ('sinai', 'mushroom')


def _reflections(s: Symbols, normal: tuple[Expression, Expression]) -> dict[Expression, Expression]:
    """Return the velocity after a specular reflection off a wall with the given normal, of any length."""
    nx, ny = normal
    projection = (s.vx * nx + s.vy * ny) / (nx**2 + ny**2)
    return {s.vx: s.vx - 2 * projection * nx, s.vy: s.vy - 2 * projection * ny}


def _sinai_laws(s: Symbols) -> list[Law]:
    """Return the free flight in a square table and the reflections off its walls and its central disc."""
    return [
        Rate(s.x, s.vx),
        Rate(s.y, s.vy),
        Event(s.x - s.size, {s.vx: -s.vx}, direction=1, label='right wall'),
        Event(s.x + s.size, {s.vx: -s.vx}, direction=-1, label='left wall'),
        Event(s.y - s.size, {s.vy: -s.vy}, direction=1, label='top wall'),
        Event(s.y + s.size, {s.vy: -s.vy}, direction=-1, label='bottom wall'),
        Event(s.x**2 + s.y**2 - s.radius**2, _reflections(s, (s.x, s.y)), direction=-1, label='disc'),
    ]


def _mushroom_laws(s: Symbols) -> list[Law]:
    """Return the free flight in a mushroom table and the reflections off the parts of its lines that are walls."""
    cap = _reflections(s, (s.x, s.y))
    in_cap, in_stem = s.y > 0, s.y < 0
    return [
        Rate(s.x, s.vx),
        Rate(s.y, s.vy),
        Event(
            s.x**2 + s.y**2 - s.radius**2,
            {s.vx: where(in_cap, cap[s.vx], s.vx), s.vy: where(in_cap, cap[s.vy], s.vy)},
            direction=1,
            label='cap',
        ),
        Event(s.y, {s.vy: where(abs(s.x) > s.width, -s.vy, s.vy)}, direction=-1, label='cap floor'),
        Event(s.x - s.width, {s.vx: where(in_stem, -s.vx, s.vx)}, direction=1, label='right stem wall'),
        Event(s.x + s.width, {s.vx: where(in_stem, -s.vx, s.vx)}, direction=-1, label='left stem wall'),
        Event(s.y + s.height, {s.vy: -s.vy}, direction=-1, label='stem floor'),
    ]


def _sinai_walls(size: float, radius: float, **_: float) -> list[np.ndarray]:
    """Return the square wall and the central disc of a Sinai table as closed polylines."""
    square = size * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]])
    angles = np.linspace(0.0, 2 * np.pi, 100)
    return [square, radius * np.column_stack([np.cos(angles), np.sin(angles)])]


SinaiBilliard = Element.define(
    'SinaiBilliard',
    states={'x': 0.1, 'y': 0.6, 'vx': 0.6, 'vy': 0.8},
    parameters={
        'size': parameter(default=1.0, domain=Domain.POSITIVE),
        'radius': parameter(default=0.4, domain=Domain.POSITIVE),
    },
    laws=_sinai_laws,
    visual=VisualSpec('bodies', {'positions': ('x', 'y'), 'trail': None, 'outline': _sinai_walls}),
)

MushroomBilliard = Element.define(
    'MushroomBilliard',
    states={'x': 0.1, 'y': 0.6, 'vx': 0.6, 'vy': 0.8},
    parameters={
        'radius': parameter(default=1.0, domain=Domain.POSITIVE),
        'width': parameter(default=0.4, domain=Domain.POSITIVE),
        'height': parameter(default=1.0, domain=Domain.POSITIVE),
    },
    laws=_mushroom_laws,
)


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
        'MushroomBilliard'
    """
    if shape == 'sinai':
        table = SinaiBilliard('particle', x=x, y=y, vx=vx, vy=vy, size=size, radius=disc_radius)
        return System(components=[table])
    if shape == 'mushroom':
        table = MushroomBilliard('particle', x=x, y=y, vx=vx, vy=vy, radius=cap_radius, width=width, height=height)
        return System(components=[table])
    error_msg = f'Unknown billiard shape {shape!r}. Use one of {", ".join(SHAPES)}.'
    raise ValidationError(error_msg)
