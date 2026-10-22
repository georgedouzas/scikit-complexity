"""Spring models: masses on springs and dampers between walls."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import cos

from ....core.base import Domain, parameter
from ....core.model import EuclideanSpace, HarmonicPotential, Interaction, LinearDrag, Particle, Rate, Spring, System
from ....core.runtime import register_model

HarmonicForce = Interaction.define(
    'HarmonicForce',
    roles=['target'],
    parameters={'F': 0.0, 'omega': parameter(default=1.0, domain=Domain.NONNEGATIVE)},
    laws=lambda s: [Rate(s.target.v, s.F * cos(s.omega * s.t) / s.target.m)],
)


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
        components=[Particle('first', x=x1, v=0.0, m=m1), Particle('second', x=x2, v=0.0, m=m2)],
        interactions=[
            HarmonicPotential('left_spring', target='first', k=k1, center=rest1),
            HarmonicPotential('right_spring', target='second', k=k2, center=rest2),
            Spring('middle_spring', first='first', second='second', k=k, length=l2),
            LinearDrag('first_damper', target='first', gamma=b1),
            LinearDrag('second_damper', target='second', gamma=b2),
            HarmonicForce('first_force', target='first', F=F1, omega=omega1),
            HarmonicForce('second_force', target='second', F=F2, omega=omega2),
        ],
        space=EuclideanSpace(1),
    )
