"""Billiard systems.

Free particles reflect off the walls of Sinai and mushroom tables and off three open discs.

`ANGLES` holds the polar angles of the centers of the three discs, one below the origin and two above it. `WALLS`
holds the walls of every table.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import EventLaw, RateLaw
from ....core.components.laws.utils import where
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Role

ANGLES = (-math.pi / 2, math.pi / 6, 5 * math.pi / 6)
SHAPES = ('sinai', 'mushroom', 'three_disk')
WALLS = {
    'sinai': (
        ('right_wall', 1),
        ('left_wall', -1),
        ('top_wall', 1),
        ('bottom_wall', -1),
        ('disc', -1),
    ),
    'mushroom': (
        ('cap', 1),
        ('cap_floor', -1),
        ('right_stem_wall', 1),
        ('left_stem_wall', -1),
        ('stem_floor', -1),
    ),
    'three_disk': (('disc_0', -1), ('disc_1', -1), ('disc_2', -1)),
}


class MushroomBilliardElement(Element):
    """Free particle in a mushroom table.

    The table is a half disc cap of radius `radius` on a stem of half width `width`.

    The particle flies straight, and reflects specularly off the cap, the floor of the cap beside the stem, the walls of
    the stem, and its floor at depth `height`. Its phase space mixes regular orbits that stay in the cap with chaotic
    orbits that enter the stem.

    Args:
        label:
            The label of the particle.

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
            How the particle is drawn by default, the `visual` of its class.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import MushroomBilliardElement
        >>> MushroomBilliardElement('ball').width
        0.4
    """

    def __init__(
        self,
        label: str,
        *,
        x: float = 0.1,
        y: float = 0.6,
        vx: float = 0.6,
        vy: float = 0.8,
        radius: Annotated[float, RealSet.open(0, oo)] = 1.0,
        width: Annotated[float, RealSet.open(0, oo)] = 0.4,
        height: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.width = width
        self.height = height


class SinaiBilliardElement(Element):
    """Free particle in the Sinai billiard.

    The table is a square of half side `half_side` with a central disc of radius `radius`.

    The particle flies straight, and its specular reflections off the four walls and the disc make its motion ergodic
    and chaotic.

    Args:
        label:
            The label of the particle.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        vx:
            The initial first velocity.

        vy:
            The initial second velocity.

        half_side:
            The half side of the square.

        radius:
            The radius of the disc.

    Attributes:
        default_visual_:
            How the particle is drawn by default, the `visual` of its class.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import SinaiBilliardElement
        >>> SinaiBilliardElement('ball').radius
        0.4
    """

    @staticmethod
    def _get_sinai_walls(half_side: float, radius: float, **_: float) -> list[np.ndarray]:
        """Return the square wall and central disc of a Sinai table as closed polylines."""
        square = half_side * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]])
        angles = np.linspace(0.0, 2 * np.pi, 100)
        return [square, radius * np.column_stack([np.cos(angles), np.sin(angles)])]

    visual = VisualSpec(
        'bodies',
        {'positions': ('x', 'y'), 'trail': None, 'outline': _get_sinai_walls},
    )

    def __init__(
        self,
        label: str,
        *,
        x: float = 0.1,
        y: float = 0.6,
        vx: float = 0.6,
        vy: float = 0.8,
        half_side: Annotated[float, RealSet.open(0, oo)] = 1.0,
        radius: Annotated[float, RealSet.open(0, oo)] = 0.4,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.half_side = half_side
        self.radius = radius


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

    @staticmethod
    def _get_three_disk_walls(spacing: float, radius: float, **_: float) -> list[np.ndarray]:
        """Return the three discs of an open three-disk billiard as closed polylines."""
        angles = np.linspace(0.0, 2 * np.pi, 100)
        circle = radius * np.column_stack([np.cos(angles), np.sin(angles)])
        distance = spacing / math.sqrt(3)
        return [circle + distance * np.array([math.cos(angle), math.sin(angle)]) for angle in ANGLES]

    visual = VisualSpec(
        'bodies',
        {'positions': ('x', 'y'), 'trail': None, 'outline': _get_three_disk_walls},
    )

    def __init__(
        self,
        label: str,
        *,
        x: float = 0.1,
        y: float = 0.6,
        vx: float = 0.6,
        vy: float = 0.8,
        spacing: Annotated[float, RealSet.open(0, oo)] = 2.4,
        radius: Annotated[float, RealSet.open(0, oo)] = 0.4,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.spacing = spacing
        self.radius = radius


class FreeFlightInteraction(Interaction):
    """Free flight `x' = vx`, `y' = vy` of a particle in the plane.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import FreeFlightInteraction
        >>> FreeFlightInteraction('flight', target='particle').target
        'particle'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the position of the particle.

        Args:
            context:
                The namespace of the particle and the time `t`.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            p, t = context.target, context.t
            self.laws_ = [RateLaw(p.x.diff(t), p.vx), RateLaw(p.y.diff(t), p.vy)]


class BilliardWallInteraction(Interaction):
    """Specular reflection of a particle off a wall of a billiard table.

    When the guard `g` of the wall crosses zero in the given direction, the velocity `v = (vx, vy)` of the particle
    becomes `v - 2 (v . n) n / |n|²`, with the normal `n` the gradient of `g` in the position `(x, y)`. The walls of the
    tables read the sizes of the particle's table, and some apply only where a condition holds:

    - `'right_wall'`, `'left_wall'`, `'top_wall'`, and `'bottom_wall'` are the sides `x = ±half_side` and `y =
      ±half_side` of the Sinai square, and `'disc'` its central disc of radius `radius`.
    - `'cap'` is the circle of radius `radius` above `y = 0`, `'cap_floor'` the line `y = 0` beside the stem,
      `'right_stem_wall'` and `'left_stem_wall'` the lines `x = ±width` below it, and `'stem_floor'` the line `y =
      -height` of the mushroom.
    - `'disc_0'`, `'disc_1'`, and `'disc_2'` are the discs of radius `radius` of the three-disk table, centered at the
      distance `spacing / √3` from the origin at the polar angles of `ANGLES`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

        wall:
            The name of the wall.

        direction:
            The direction in which the guard crosses zero, `1` upward, `-1` downward, and `0` either way.

    Raises:
        ValidationError:
            The wall is unknown.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import BilliardWallInteraction
        >>> BilliardWallInteraction('cap', target='particle', wall='cap', direction=1).wall
        'cap'
    """

    law_classes = (EventLaw,)

    def __init__(self, label: str, *, target: Role, wall: str = 'disc', direction: int = 0) -> None:
        super().__init__(label)
        self.target = target
        self.wall = wall
        self.direction = direction

    def _get_guard(self, p: SimpleNamespace) -> Expression:
        """Return the guard of the wall, an expression of the position that is zero on the wall."""
        if self.wall.startswith('disc_'):
            angle = ANGLES[int(self.wall.removeprefix('disc_'))]
            distance = p.spacing / math.sqrt(3)
            return (p.x - distance * math.cos(angle)) ** 2 + (p.y - distance * math.sin(angle)) ** 2 - p.radius**2
        guards = {
            'right_wall': lambda: p.x - p.half_side,
            'left_wall': lambda: p.x + p.half_side,
            'top_wall': lambda: p.y - p.half_side,
            'bottom_wall': lambda: p.y + p.half_side,
            'disc': lambda: p.x**2 + p.y**2 - p.radius**2,
            'cap': lambda: p.x**2 + p.y**2 - p.radius**2,
            'cap_floor': lambda: p.y,
            'right_stem_wall': lambda: p.x - p.width,
            'left_stem_wall': lambda: p.x + p.width,
            'stem_floor': lambda: p.y + p.height,
        }
        return SR(guards[self.wall]())

    def _get_region(self, p: SimpleNamespace) -> object:
        """Return the condition under which the surface reflects, or `None` when all of it does."""
        regions = {
            'cap': lambda: p.y > 0,
            'cap_floor': lambda: abs(p.x) > p.width,
            'right_stem_wall': lambda: p.y < 0,
            'left_stem_wall': lambda: p.y < 0,
        }
        return regions[self.wall]() if self.wall in regions else None

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the event that reflects the velocity about the normal of the wall.

        Args:
            context:
                The namespace of the particle and the time `t`.

        Raises:
            ValidationError:
                The wall is unknown.
        """
        super()._prepare(context)
        known = [name for walls in WALLS.values() for name, _ in walls]
        if self.wall not in known:
            error_msg = f'Unknown billiard wall {self.wall!r}. Use one of {", ".join(known)}.'
            raise ValidationError(error_msg)
        if isinstance(context, SimpleNamespace):
            p = context.target
            x, y = SR.symbol('_x'), SR.symbol('_y')
            guard = self._get_guard(p)
            flat = guard.subs({p.x: x, p.y: y})
            normal = [SR(flat.diff(x)).subs({x: p.x, y: p.y}), SR(flat.diff(y)).subs({x: p.x, y: p.y})]
            velocities = [p.vx, p.vy]
            speed = sum((v * n for v, n in zip(velocities, normal, strict=True)), SR(0))
            scale = 2 * speed / sum((n**2 for n in normal), SR(0))
            reset = {v: v - scale * n for v, n in zip(velocities, normal, strict=True)}
            region = self._get_region(p)
            if region is not None:
                reset = {v: where(region, value, v) for v, value in reset.items()}
            self.laws_ = [EventLaw(guard, reset, self.direction, label=self.label)]


class BilliardsSystem(BaseSystem):
    """A particle moving freely on a billiard table.

    The particle reflects specularly off the walls of the table.

    The Sinai table is the square `[-half_side, half_side]²` with a disc of radius `disc_radius` at its center, and its
    motion is chaotic. The mushroom table is the upper half of a disc of radius `cap_radius` on a stem of half width
    `width` and height `height`, and its phase space mixes regular orbits that stay in the cap with chaotic orbits that
    enter the stem. The three-disk table is open: three discs of radius `disc_radius` sit at the corners of an
    equilateral triangle of side `spacing` centered at the origin, with one disc below the origin. A particle that comes
    from far away is scattered chaotically between the discs and leaves again, and the number of its collisions and its
    outgoing angle depend on its incoming direction through a fractal scattering function. Every bounce is an event, and
    the speed is conserved. The system has the position `particle.x`, `particle.y` and the velocity `particle.vx`,
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

        half_side:
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        half_side: float = 1.0,
        disc_radius: float = 0.4,
        cap_radius: float = 1.0,
        width: float = 0.4,
        height: float = 1.0,
        spacing: float = 2.4,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.half_side = half_side
        self.disc_radius = disc_radius
        self.cap_radius = cap_radius
        self.width = width
        self.height = height
        self.spacing = spacing

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a particle moving freely on a billiard table.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.

        Raises:
            ValidationError:
                The shape is unknown.
        """
        super()._prepare(context)
        start = {'x': self.x, 'y': self.y, 'vx': self.vx, 'vy': self.vy}
        table: Element
        if self.shape == 'sinai':
            table = SinaiBilliardElement('particle', **start, half_side=self.half_side, radius=self.disc_radius)
        elif self.shape == 'mushroom':
            table = MushroomBilliardElement(
                'particle',
                **start,
                radius=self.cap_radius,
                width=self.width,
                height=self.height,
            )
        elif self.shape == 'three_disk':
            table = ThreeDiskBilliardElement('particle', **start, spacing=self.spacing, radius=self.disc_radius)
        else:
            error_msg = f'Unknown billiard shape {self.shape!r}. Use one of {", ".join(SHAPES)}.'
            raise ValidationError(error_msg)
        walls = [
            BilliardWallInteraction(name, target='particle', wall=name, direction=direction)
            for name, direction in WALLS[self.shape]
        ]
        flight = FreeFlightInteraction('flight', target='particle')
        self.system_ = System(self.label, elements=[table], interactions=[flight, *walls])
