"""Spring models: masses on springs and dampers between walls."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, oo
from sage.sets.real_set import RealSet

from ....core.elements import ParticleElement
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RateLaw
from ....core.spaces import ContinuousSpace
from ....core.system import System
from ....specs import ParameterSpec, RoleSpec
from ..._catalog import register_model
from ..common.particles import HarmonicPotentialInteraction, LinearDragInteraction, SpringInteraction


class HarmonicForceInteraction(Interaction):
    """Harmonic force interaction."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[object, RoleSpec()],
        F: Annotated[object, ParameterSpec()] = 0.0,
        omega: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.F = F
        self.omega = omega

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the harmonic force interaction."""
        return [RateLaw(s.target.v, s.F * cos(s.omega * s.t) / s.target.m)]


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'oscillation', 'normal modes', 'forcing'),
    references=('Ryzhov et al. 2021',),
    defaults={'until': 30.0, 'dt': 0.05},
)
def two_mass_spring_damper(
    x1: float = 1.2,
    x2: float = 2.0,
    *,
    m1: float = 1.0,
    m2: float = 1.0,
    k1: float = 1.0,
    k2: float = 1.0,
    k: float = 1.0,
    b1: float = 0.0,
    b2: float = 0.0,
    l1: float = 1.0,
    l2: float = 1.0,
    F1: float = 0.0,
    F2: float = 0.0,
    omega1: float = 1.0,
    omega2: float = 1.0,
) -> System:
    """Return two masses between two walls, tied to the walls and to each other by springs, with dampers and forces.

    The first mass rests at `x01 = l1` and the second at `x02 = l1 + l2`. They follow
    `m1 x1'' = -k1 (x1 - x01) + k ((x2 - x1) - (x02 - x01)) - b1 x1' + F1 cos(ω1 t)` and
    `m2 x2'' = -k ((x2 - x1) - (x02 - x01)) - k2 (x2 - x02) - b2 x2' + F2 cos(ω2 t)`.

    Args:
        x1:
            The initial position of the first mass.

        x2:
            The initial position of the second mass.

        m1:
            The first mass.

        m2:
            The second mass.

        k1:
            The stiffness of the spring between the left wall and the first mass.

        k2:
            The stiffness of the spring between the second mass and the right wall.

        k:
            The stiffness of the spring between the masses.

        b1:
            The damping coefficient of the first mass.

        b2:
            The damping coefficient of the second mass.

        l1:
            The rest length of the first spring.

        l2:
            The rest length of the middle spring.

        F1:
            The amplitude of the periodic force on the first mass.

        F2:
            The amplitude of the periodic force on the second mass.

        omega1:
            The angular frequency of the force on the first mass.

        omega2:
            The angular frequency of the force on the second mass.

    Returns:
        system:
            The model, with the masses `first` and `second` and their positions `first.x` and `second.x`.

    Examples:
        >>> from skcomplex.models.physics.mechanics import two_mass_spring_damper
        >>> two_mass_spring_damper(l1=2.0).interactions[0].center
        2.0
    """
    rest1, rest2 = l1, l1 + l2
    return System(
        elements=[ParticleElement('first', x=x1, v=0.0, m=m1), ParticleElement('second', x=x2, v=0.0, m=m2)],
        interactions=[
            HarmonicPotentialInteraction('left_spring', target='first', k=k1, center=rest1),
            HarmonicPotentialInteraction('right_spring', target='second', k=k2, center=rest2),
            SpringInteraction('middle_spring', first='first', second='second', k=k, length=l2),
            LinearDragInteraction('first_damper', target='first', gamma=b1),
            LinearDragInteraction('second_damper', target='second', gamma=b2),
            HarmonicForceInteraction('first_force', target='first', F=F1, omega=omega1),
            HarmonicForceInteraction('second_force', target='second', F=F2, omega=omega2),
        ],
        space=ContinuousSpace(n_dim=1),
    )
