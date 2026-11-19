"""Define the motion of mechanical coordinates from a Lagrangian.

The Euler-Lagrange equations give the accelerations of the coordinates.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from abc import abstractmethod
from types import SimpleNamespace

from sage.all__sagemath_symbolics import SR, Expression, matrix, vector

from .....core.components.interactions import Interaction
from .....core.components.laws import RateLaw
from .....core.components.spaces import BaseSpace


class LagrangianInteraction(Interaction):
    """Motion of mechanical coordinates from the Euler-Lagrange equations of a Lagrangian `L`.

    Every coordinate `q` follows `d/dt (dL/dq') - dL/dq + dD/dq' = Q`, with the Rayleigh dissipation `D` and the
    generalized force `Q`. The equations are linear in the accelerations, so the interaction solves them and gives the
    acceleration of every coordinate. A subclass returns its coordinates from `_get_coordinates`, its Lagrangian from
    `_get_lagrangian`, and may return a dissipation from `_get_dissipation` and forces from `_get_forces`. It is the
    swing of a pendulum, the precession of a gyroscope, or the sway of a crane with its load.

    Args:
        label:
            The label of the interaction.

    Examples:
        >>> from sage.all__sagemath_symbolics import cos
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.types import Role
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.mechanics import LagrangianInteraction
        >>> class SwingInteraction(LagrangianInteraction):
        ...     def __init__(self, label, *, target: Role):
        ...         super().__init__(label)
        ...         self.target = target
        ...
        ...     def _get_coordinates(self, context):
        ...         return [context.target.q]
        ...
        ...     def _get_lagrangian(self, context, positions, velocities):
        ...         return velocities[0] ** 2 / 2 + cos(positions[0])
        >>> bob = Element('bob', values={'q': 0.1})
        >>> system = System(elements=[bob], interactions=[SwingInteraction('swing', target='bob')]).prepare()
        >>> round(float(system.simulate(until=3.2, dt=0.01).get_values('bob.q')[-1]), 2)
        -0.1
    """

    law_classes = (RateLaw,)

    @abstractmethod
    def _get_coordinates(self, context: SimpleNamespace) -> list[Expression]:
        """Return the coordinates, each a state of a role as a function of time.

        Args:
            context:
                The namespace of the roles, the time `t`, and the parameters.

        Returns:
            coordinates:
                The coordinates.
        """

    @abstractmethod
    def _get_lagrangian(
        self,
        context: SimpleNamespace,
        positions: list[Expression],
        velocities: list[Expression],
    ) -> Expression:
        """Return the Lagrangian of the positions and the velocities of the coordinates.

        Args:
            context:
                The namespace of the roles, the time `t`, and the parameters.

            positions:
                A symbol for the position of every coordinate.

            velocities:
                A symbol for the velocity of every coordinate.

        Returns:
            lagrangian:
                The Lagrangian.
        """

    def _get_dissipation(
        self,
        context: SimpleNamespace,
        positions: list[Expression],
        velocities: list[Expression],
    ) -> Expression:
        """Return the Rayleigh dissipation, zero by default.

        Args:
            context:
                The namespace of the roles, the time `t`, and the parameters.

            positions:
                A symbol for the position of every coordinate.

            velocities:
                A symbol for the velocity of every coordinate.

        Returns:
            dissipation:
                The dissipation.
        """
        return SR(0)

    def _get_forces(
        self,
        context: SimpleNamespace,
        positions: list[Expression],
        velocities: list[Expression],
    ) -> list[Expression]:
        """Return the generalized force on every coordinate, zero by default.

        Args:
            context:
                The namespace of the roles, the time `t`, and the parameters.

            positions:
                A symbol for the position of every coordinate.

            velocities:
                A symbol for the velocity of every coordinate.

        Returns:
            forces:
                The generalized forces.
        """
        return [SR(0)] * len(positions)

    @staticmethod
    def _get_frozen(context: SimpleNamespace) -> dict[Expression, Expression]:
        """Return a symbol for every value of a role, which the derivative in time then leaves alone.

        The values of the elements are functions of time until the laws fix which of them are states, so only the
        coordinates, given as symbols, and the time itself may change with time in the equations.
        """
        frozen: dict[Expression, Expression] = {}
        for role in vars(context).values():
            if not isinstance(role, SimpleNamespace):
                continue
            for value in vars(role).values():
                if isinstance(value, Expression) and value.has(context.t):
                    frozen[value] = SR.symbol(f'_frozen_{len(frozen)}')
        return frozen

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of every coordinate from the Euler-Lagrange equations.

        Args:
            context:
                The namespace of the roles, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            t = context.t
            coordinates = self._get_coordinates(context)
            count = len(coordinates)
            positions = [SR.symbol(f'_position_{index}') for index in range(count)]
            velocities = [SR.symbol(f'_velocity_{index}') for index in range(count)]
            accelerations = [SR.symbol(f'_acceleration_{index}') for index in range(count)]
            frozen = self._get_frozen(context)
            lagrangian = SR(self._get_lagrangian(context, positions, velocities)).subs(frozen)
            dissipation = SR(self._get_dissipation(context, positions, velocities)).subs(frozen)
            forces = [SR(force).subs(frozen) for force in self._get_forces(context, positions, velocities)]
            equations = []
            for position, velocity, force in zip(positions, velocities, forces, strict=True):
                momentum = lagrangian.diff(velocity)
                change = momentum.diff(t) + sum(
                    momentum.diff(other) * speed + momentum.diff(speed) * acceleration
                    for other, speed, acceleration in zip(positions, velocities, accelerations, strict=True)
                )
                equations.append(change - lagrangian.diff(position) + dissipation.diff(velocity) - force)
            inertia = matrix(SR, [[equation.diff(item) for item in accelerations] for equation in equations])
            rest = vector(SR, [equation.subs(dict.fromkeys(accelerations, 0)) for equation in equations])
            solution = inertia.solve_right(-rest)
            back = {
                **{symbol: value for value, symbol in frozen.items()},
                **dict(zip(positions, coordinates, strict=True)),
                **{velocity: item.diff(t) for velocity, item in zip(velocities, coordinates, strict=True)},
            }
            self.laws_ = [
                RateLaw(item.diff(t, 2), value.subs(back)) for item, value in zip(coordinates, solution, strict=True)
            ]
