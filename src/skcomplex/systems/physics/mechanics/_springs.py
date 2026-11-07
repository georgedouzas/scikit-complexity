"""Spring systems: masses on springs and dampers between walls."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.components.spaces import ContinuousSpace
from ....core.components.system import BaseSystem, System
from ..common.particles import (
    DampingInteraction,
    HarmonicForceInteraction,
    HarmonicPotentialInteraction,
    MassPointElement,
    SpringInteraction,
)


class TwoMassSpringDamperSystem(BaseSystem):
    """Two masses between two walls, tied to the walls and to each other by springs, with dampers and forces.

    The first mass rests at `x01 = l1` and the second at `x02 = l1 + l2`. They follow `m1 x1'' = -k1 (x1 - x01) + k ((x2
    - x1) - (x02 - x01)) - b1 x1' + F1 cos(ω1 t)` and `m2 x2'' = -k ((x2 - x1) - (x02 - x01)) - k2 (x2 - x02) - b2 x2' +
    F2 cos(ω2 t)`. The system has the masses `first` and `second` and their positions `first.x` and `second.x`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import TwoMassSpringDamperSystem
        >>> TwoMassSpringDamperSystem(l1=2.0).prepare().system_.interactions[0].center
        2.0
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'ode', 'oscillation', 'normal modes', 'forcing')
    references = ('Ryzhov et al. 2021',)
    until = 30.0
    dt = 0.05

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x1 = x1
        self.x2 = x2
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2
        self.k = k
        self.b1 = b1
        self.b2 = b2
        self.l1 = l1
        self.l2 = l2
        self.F1 = F1
        self.F2 = F2
        self.omega1 = omega1
        self.omega2 = omega2

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        rest1, rest2 = self.l1, self.l1 + self.l2
        return System(
            self.label,
            elements=[
                MassPointElement('first', x=self.x1, x_d1=0.0, m=self.m1),
                MassPointElement('second', x=self.x2, x_d1=0.0, m=self.m2),
            ],
            interactions=[
                HarmonicPotentialInteraction('left_spring', target='first', k=self.k1, center=rest1),
                HarmonicPotentialInteraction('right_spring', target='second', k=self.k2, center=rest2),
                SpringInteraction('middle_spring', first='first', second='second', k=self.k, length=self.l2),
                DampingInteraction('first_damper', target='first', gamma=self.b1),
                DampingInteraction('second_damper', target='second', gamma=self.b2),
                HarmonicForceInteraction('first_force', target='first', F=self.F1, omega=self.omega1),
                HarmonicForceInteraction('second_force', target='second', F=self.F2, omega=self.omega2),
            ],
            space=ContinuousSpace(n_dim=1),
        )
