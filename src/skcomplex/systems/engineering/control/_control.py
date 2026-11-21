"""Control systems: feedback loops that keep or drive a system towards a goal."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import Expression, cos, oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, ConstraintLaw, EventLaw
from ....core.components.laws.utils import clip, sign, step, where
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, RK4Solver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations
from ..common.electrical import BranchInteraction, GroundInteraction, VoltageSourceInteraction

DISTURBANCE_TIME = 10.0
CHANGE_TIME = 20.0
HOURS_PER_DAY = 24.0
POSITIVE = RealSet.open(0, oo)


class HeatCapacityInteraction(Interaction):
    """Heat capacity `C` of a node of temperature `T`, which stores the heat `C T'` that flows into the node.

    The heat flows of the components that join a node add into its balance, which is zero, so with a capacity the node
    warms at `T' = Q / C` under the net heat flow `Q` into it.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        node:
            The path of the node, an element with the temperature `T`.

        C:
            The heat capacity.

    Examples:
        >>> from skcomplex.systems.engineering.control import HeatCapacityInteraction
        >>> HeatCapacityInteraction('air', node='room', C=2e5).C
        200000.0
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (ConstraintLaw,)

    def __init__(self, label: str, *, node: Role, C: Annotated[float, POSITIVE] = 1.0) -> None:
        super().__init__(label)
        self.node = node
        self.C = C

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the heat the capacity stores as a term of the balance of the node."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            T = context.node.T
            self.laws_: list[BaseLaw] = [ConstraintLaw(-context.C * T.diff(context.t), 0, target=T)]


class ThermalConductorInteraction(Interaction):
    """Conductor of the heat flow `G (a.T - b.T)` from the node `a` to the node `b`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        a:
            The path of the node the heat leaves.

        b:
            The path of the node the heat enters.

        G:
            The thermal conductance.

    Examples:
        >>> from skcomplex.systems.engineering.control import ThermalConductorInteraction
        >>> ThermalConductorInteraction('wall', a='room', b='outside', G=2.0).G
        2.0
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (ConstraintLaw,)

    def __init__(self, label: str, *, a: Role, b: Role, G: Annotated[float, POSITIVE] = 1.0) -> None:
        super().__init__(label)
        self.a = a
        self.b = b
        self.G = G

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the heat flow as a term of the balances of both nodes."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            a, b = context.a.T, context.b.T
            flow = context.G * (a - b)
            self.laws_: list[BaseLaw] = [ConstraintLaw(-flow, 0, target=a), ConstraintLaw(flow, 0, target=b)]


class ConvectionInteraction(Interaction):
    """Surface of a node that loses the heat flow `h (T - fluid)` to a fluid of a fixed temperature.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        node:
            The path of the node, an element with the temperature `T`.

        h:
            The convective conductance of the surface.

        fluid:
            The temperature of the fluid.

    Examples:
        >>> from skcomplex.systems.engineering.control import ConvectionInteraction
        >>> ConvectionInteraction('outside', node='wall', fluid=5.0).fluid
        5.0
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (ConstraintLaw,)

    def __init__(
        self,
        label: str,
        *,
        node: Role,
        h: Annotated[float, POSITIVE] = 1.0,
        fluid: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.node = node
        self.h = h
        self.fluid = fluid

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the heat lost to the fluid as a term of the balance of the node."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            T = context.node.T
            self.laws_: list[BaseLaw] = [ConstraintLaw(-context.h * (T - context.fluid), 0, target=T)]


class HeatingResistorInteraction(BranchInteraction):
    """Resistor between the electrical nodes `a` and `b` that heats the thermal node `heated`.

    It conducts while the value `closed` of the element `switch` is 1. Its current is `closed (a.v - b.v) / R`, and
    the power it dissipates, the voltage times the current, flows into the heated node.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        a:
            The path of the electrical node the current leaves.

        b:
            The path of the electrical node the current enters.

        heated:
            The path of the thermal node the heat enters, an element with the temperature `T`.

        switch:
            The path of the element that holds the switch `closed`, 1 for closed and 0 for open.

        R:
            The resistance.

    Examples:
        >>> from skcomplex.systems.engineering.control import HeatingResistorInteraction
        >>> HeatingResistorInteraction('heating', a='supply', b='neutral', heated='room', switch='heater', R=25.0).R
        25.0
    """

    def __init__(
        self,
        label: str,
        *,
        a: Role,
        b: Role,
        heated: Role,
        switch: Role,
        R: Annotated[float, POSITIVE] = 1.0,
    ) -> None:
        super().__init__(label, a=a, b=b)
        self.heated = heated
        self.switch = switch
        self.R = R

    def _get_branch_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the switched current into both electrical balances and the power into the thermal balance."""
        voltage = s.a.v - s.b.v
        current = s.switch.closed * voltage / s.R
        return [
            ConstraintLaw(-current, 0, target=s.a.v),
            ConstraintLaw(current, 0, target=s.b.v),
            ConstraintLaw(voltage * current, 0, target=s.heated.T),
        ]


class SwitchInteraction(Interaction):
    """Switch that opens or closes when the temperature of a sensor crosses a threshold.

    When the temperature `T` of `sensor` crosses `threshold` in `direction`, the value `closed` of `switch` becomes
    `value`, as a thermostat switches a heater or a pressure switch a pump.

    Args:
        label:
            The label of the interaction, which names its events.

        sensor:
            The path of the element whose temperature `T` the switch watches.

        switch:
            The path of the element that holds the switch `closed`.

        threshold:
            The temperature at which the switch acts.

        value:
            The value the switch takes, 1 for closed and 0 for open.

        direction:
            The direction of the crossing, 1 when the temperature rises and -1 when it falls.

    Examples:
        >>> from skcomplex.systems.engineering.control import SwitchInteraction
        >>> SwitchInteraction('open_switch', sensor='room', switch='heater', threshold=20.5, value=0.0).direction
        1
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (EventLaw,)

    def __init__(
        self,
        label: str,
        *,
        sensor: Role,
        switch: Role,
        threshold: float = 0.0,
        value: float = 0.0,
        direction: int = 1,
    ) -> None:
        super().__init__(label)
        self.sensor = sensor
        self.switch = switch
        self.threshold = threshold
        self.value = value
        self.direction = direction

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the crossing of the threshold and the value of the switch it sets."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            guard = context.sensor.T - context.threshold
            self.laws_: list[BaseLaw] = [
                EventLaw(guard, {context.switch.closed: context.value}, direction=self.direction),
            ]


class ThermostatSystem(DynamicalSystem):
    """A room heated by a furnace under a thermostat while heat leaks to the outside.

    The furnace adds the gap between the setting and the room temperature, at most `furnace` degrees per hour, and
    nothing when the room is warmer than the setting. The room loses `leak` of its difference to the outside temperature
    per hour, so it settles below the setting, at `(setting + leak * outside) / (1 + leak)` for a constant outside. The
    outside temperature swings by `swing` around `outside` over a day, warmest at noon. The system has the room
    temperature `room.T` in hours.

    Args:
        T:
            The initial room temperature.

        setting:
            The thermostat setting.

        furnace:
            The largest heating rate of the furnace, in degrees per hour.

        leak:
            The fraction of the difference to the outside lost per hour.

        outside:
            The mean outside temperature.

        swing:
            The amplitude of the daily swing of the outside temperature.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.control import ThermostatSystem
        >>> ThermostatSystem(leak=0.3).prepare().system_.elements[0].values['leak']
        0.3
    """

    field = 'engineering/control'
    topics = ('control', 'feedback', 'stocks and flows', 'ode')
    references = ('Meadows 2008',)
    until = 24.0
    dt = 0.25

    def __init__(
        self,
        T: float = 10.0,
        *,
        setting: float = 18.0,
        furnace: float = 5.0,
        leak: float = 0.1,
        outside: float = 10.0,
        swing: float = 0.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.T = T
        self.setting = setting
        self.furnace = furnace
        self.leak = leak
        self.outside = outside
        self.swing = swing

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'room'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state by name."""
        return {
            'T': self.T,
        }

    def _get_allowed(self) -> dict[str, object]:
        """Return the allowed values of the states and parameters that have them, by name."""
        return {
            'furnace': RealSet.closed_open(0, oo),
            'leak': RealSet.closed_open(0, oo),
            'swing': RealSet.closed_open(0, oo),
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the value of every parameter by name."""
        return {
            'setting': self.setting,
            'furnace': self.furnace,
            'leak': self.leak,
            'outside': self.outside,
            'swing': self.swing,
        }

    def _get_equations(self) -> Equations:
        """Return the saturated heating and the leak of the temperature `T`."""
        return {
            'T': lambda s: (
                clip(s.target.setting - s.target.T, 0, s.target.furnace)
                - s.target.leak
                * (
                    s.target.T
                    - s.target.outside
                    - s.target.swing * cos(2 * pi * (s.t - HOURS_PER_DAY / 2) / HOURS_PER_DAY)
                )
            ),
        }


class PidLoopSystem(DynamicalSystem):
    """A work process whose output a PID controller keeps at a set point.

    The controller acts against a disturbance. The output follows the first-order process `τ y' = -y + gain u + d(t)`,
    and the controller sets `u = Kp e + Ki ∫ e dt - Kd y'` from the error `e = r(t) - y`, with the derivative taken on
    the output, so `(τ + gain Kd) y' = -y + gain (Kp e + Ki ∫ e dt) + d(t)`. The disturbance `d` steps in at time 10,
    and a coordinator moves the set point `r` by `change` at time 20. The integral action removes the error that the
    disturbance leaves under proportional control alone. The system has the output `process.y` and the integral of the
    error `process.integral`.

    Args:
        y:
            The initial output.

        tau:
            The time constant of the process.

        gain:
            The gain of the actuator.

        Kp:
            The proportional gain.

        Ki:
            The integral gain.

        Kd:
            The derivative gain.

        setpoint:
            The initial set point.

        change:
            The change of the set point at time 20.

        disturbance:
            The size of the disturbance from time 10.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.control import PidLoopSystem
        >>> PidLoopSystem(Ki=0.0).prepare().system_.elements[0].values['Ki']
        0.0
    """

    field = 'engineering/control'
    topics = ('control', 'feedback', 'pid', 'governance', 'ode')
    references = ('Mobus 2022', 'Åström and Murray 2008')
    until = 40.0
    dt = 0.1

    def __init__(
        self,
        y: float = 0.0,
        *,
        tau: float = 1.0,
        gain: float = 1.0,
        Kp: float = 2.0,
        Ki: float = 1.0,
        Kd: float = 0.1,
        setpoint: float = 1.0,
        change: float = 0.0,
        disturbance: float = -0.5,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.y = y
        self.tau = tau
        self.gain = gain
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.change = change
        self.disturbance = disturbance

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'process'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state by name."""
        return {
            'y': self.y,
            'integral': 0.0,
        }

    def _get_allowed(self) -> dict[str, object]:
        """Return the allowed values of the states and parameters that have them, by name."""
        return {
            'tau': RealSet.open(0, oo),
            'Kp': RealSet.closed_open(0, oo),
            'Ki': RealSet.closed_open(0, oo),
            'Kd': RealSet.closed_open(0, oo),
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the value of every parameter by name."""
        return {
            'tau': self.tau,
            'gain': self.gain,
            'Kp': self.Kp,
            'Ki': self.Ki,
            'Kd': self.Kd,
            'setpoint': self.setpoint,
            'change': self.change,
            'disturbance': self.disturbance,
        }

    def _get_equations(self) -> Equations:
        """Return the rates of the output `y` and of the integral of the error."""
        return {
            'y': lambda s: (
                (
                    -s.target.y
                    + s.target.gain * (s.target.Kp * PidLoopSystem._get_error(s) + s.target.Ki * s.target.integral)
                    + s.target.disturbance * step(s.t, DISTURBANCE_TIME)
                )
                / (s.target.tau + s.target.gain * s.target.Kd)
            ),
            'integral': PidLoopSystem._get_error,
        }

    @staticmethod
    def _get_error(s: SimpleNamespace) -> Expression:
        """Return the error of the controlled process.

        It is the set point after its change less the output.
        """
        return s.target.setpoint + s.target.change * step(s.t, CHANGE_TIME) - s.target.y


class PendulumSwingUpSystem(DynamicalSystem):
    """A pendulum swung up to its inverted position.

    A bang-bang acceleration of its pivot swings it up. The pivot of an acrobat's pendulum moves horizontally with
    acceleration `f'' = u`, so the pendulum follows `θ'' + ω^2 sin θ + u cos θ = 0`. The control
    `u = -μ sign(θ' cos θ) sign(E_s - E)` pumps the energy `E = θ'^2 / 2 - ω^2 cos θ` towards `E_s = ω^2`, the energy of
    the inverted position, from below or from above. The system has the angle `pendulum.theta`, the position of the
    pivot `pendulum.f`, and their velocities.

    Args:
        theta:
            The initial angle from the hanging position.

        omega0:
            The initial angular velocity.

        omega:
            The natural frequency `sqrt(g / l)`, by default for `g = 9.8` and `l = 4`.

        mu:
            The size of the acceleration of the pivot.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.control import PendulumSwingUpSystem
        >>> PendulumSwingUpSystem(mu=0.2).prepare().system_.elements[0].values['mu']
        0.2
    """

    field = 'engineering/control'
    topics = ('control', 'bang-bang', 'pendulum', 'energy', 'ode')
    references = ('Åström and Furuta 2000', 'Dilão 2023')
    until = 100.0
    dt = 0.1

    def __init__(
        self,
        theta: float = 0.0,
        omega0: float = 0.5,
        *,
        omega: float = 1.5652475842498528,
        mu: float = 0.1,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.theta = theta
        self.omega0 = omega0
        self.omega = omega
        self.mu = mu

    def _get_solver(self) -> BaseSolver:
        """Return the solver the system is tuned to."""
        return RK4Solver(dt=0.01)

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'pendulum'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state by name."""
        return {
            'theta': self.theta,
            "theta'": self.omega0,
            'f': 0.0,
            "f'": 0.0,
        }

    def _get_allowed(self) -> dict[str, object]:
        """Return the allowed values of the states and parameters that have them, by name."""
        return {
            'omega': RealSet.open(0, oo),
            'mu': RealSet.closed_open(0, oo),
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the value of every parameter by name."""
        return {
            'omega': self.omega,
            'mu': self.mu,
        }

    def _get_equations(self) -> Equations:
        """Return the controlled rate of the angle and the rate of the control integral."""
        return {
            "theta''": lambda s: (
                -(s.target.omega**2) * sin(s.target.theta)
                - PendulumSwingUpSystem._swing_up_control(s) * cos(s.target.theta)
            ),
            "f''": PendulumSwingUpSystem._swing_up_control,
        }

    @staticmethod
    def _get_pendulum_energy(s: SimpleNamespace) -> Expression:
        """Return the energy per unit inertia of the pendulum, `θ'^2 / 2 - ω^2 cos θ`."""
        return s.target.theta.diff(s.t) ** 2 / 2 - s.target.omega**2 * cos(s.target.theta)

    @staticmethod
    def _swing_up_control(s: SimpleNamespace) -> Expression:
        """Return the bang-bang control of the energy of the pendulum.

        It is `u = -μ sign(θ' cos θ) sign(E_s - E)`.
        """
        gap = s.target.omega**2 - PendulumSwingUpSystem._get_pendulum_energy(s)
        return -s.target.mu * sign(s.target.theta.diff(s.t) * cos(s.target.theta)) * sign(gap)


def _get_energy_of_satellite(s: SimpleNamespace) -> Expression:
    """Return the total effective energy of the satellite of the system."""
    satellite = s.satellite
    return satellite.s.diff(s.t) ** 2 / 2 + satellite.Lz**2 / (2 * satellite.s**2) - 1 / satellite.s


class SatelliteTransferSystem(DynamicalSystem):
    """A satellite moved between Kepler orbits by radial thrust.

    The orbits have equal angular momentum. In normalised polar coordinates the satellite follows
    `s'' = L_z^2 / s^3 - 1 / s^2 - ε sigma` and `θ' = L_z / s^2`. Radial thrust keeps the angular momentum and changes
    the effective energy `H = s'^2 / 2 + L_z^2 / (2 s^2) - 1 / s` at the rate `-ε sigma s'`, so the controller pushes
    inwards, `sigma = 1`, while the satellite falls and outwards, `sigma = -1`, while it rises, until `H` reaches the
    final energy. From the periapsis of the orbit with `H = -0.7`, the transfer to `H = -0.2` takes about 9.81.

    The system has the distance `satellite.s`, the radial velocity `satellite.s'`, the angle `satellite.theta`, and
    the observable `energy`.

    Args:
        s:
            The initial distance to the planet.

        velocity:
            The initial radial velocity.

        theta:
            The initial angle.

        Lz:
            The angular momentum.

        epsilon:
            The thrust.

        H_final:
            The effective energy of the final orbit.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.control import SatelliteTransferSystem
        >>> SatelliteTransferSystem().prepare().system_.elements[0].values['Lz']
        0.7
    """

    field = 'engineering/control'
    topics = ('control', 'orbits', 'celestial mechanics', 'ode')
    references = ('Dilão et al. 2022', 'Dilão 2023')
    until = 20.0
    dt = 0.05
    observables = (Observable('energy', _get_energy_of_satellite),)

    def __init__(
        self,
        s: float = 0.314,
        velocity: float = 0.0,
        theta: float = 0.0,
        *,
        Lz: float = 0.7,
        epsilon: float = 0.1,
        H_final: float = -0.2,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.s = s
        self.velocity = velocity
        self.theta = theta
        self.Lz = Lz
        self.epsilon = epsilon
        self.H_final = H_final

    def _get_solver(self) -> BaseSolver:
        """Return the solver the system is tuned to."""
        return RK4Solver(dt=0.005)

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'satellite'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state by name."""
        return {
            's': self.s,
            "s'": self.velocity,
            'theta': self.theta,
        }

    def _get_allowed(self) -> dict[str, object]:
        """Return the allowed values of the states and parameters that have them, by name."""
        return {
            's': RealSet.open(0, oo),
            'Lz': RealSet.open(0, oo),
            'epsilon': RealSet.closed_open(0, oo),
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the value of every parameter by name."""
        return {
            'Lz': self.Lz,
            'epsilon': self.epsilon,
            'H_final': self.H_final,
        }

    def _get_equations(self) -> Equations:
        """Return the radial rate under thrust and the rate of the angle."""
        return {
            "s''": lambda s: (
                s.target.Lz**2 / s.target.s**3
                - 1 / s.target.s**2
                - s.target.epsilon * SatelliteTransferSystem._get_thrust_direction(s)
            ),
            'theta': lambda s: s.target.Lz / s.target.s**2,
        }

    @staticmethod
    def _get_satellite_energy(s: SimpleNamespace) -> Expression:
        """Return the total effective energy of the satellite.

        It is `s'^2 / 2 + L_z^2 / (2 s^2) - 1 / s`.
        """
        return s.target.s.diff(s.t) ** 2 / 2 + s.target.Lz**2 / (2 * s.target.s**2) - 1 / s.target.s

    @staticmethod
    def _get_thrust_direction(s: SimpleNamespace) -> Expression:
        """Return the thrust `sigma`.

        It points inwards while the satellite falls, outwards while it rises, to the end energy.
        """
        return where(
            SatelliteTransferSystem._get_satellite_energy(s) < s.target.H_final,
            where(s.target.s.diff(s.t) < 0, 1, -1),
            0,
        )


def _get_heating_power(s: SimpleNamespace) -> Expression:
    """Return the power that the source delivers to the heater."""
    power: Expression = -(s.supply.v - s.neutral.v) * s.source.i
    return power


class HeatedRoomSystem(BaseSystem):
    """A room assembled from thermal and electrical components.

    It is heated under a thermostat with hysteresis. A voltage source drives a heating resistor whose heat enters the
    room, and the room loses heat through a wall of its own heat capacity to the outside air. The thermostat opens the
    switch of the resistor when the room rises through `setting + band / 2` and closes it when the room falls through
    `setting - band / 2`. The system has the room temperature `room.T` and the switch `heater.closed`, in seconds.

    Args:
        T:
            The initial temperature of the room.

        setting:
            The temperature the thermostat keeps.

        band:
            The width of the hysteresis band around the setting.

        outside:
            The temperature of the outside air, which is also the initial temperature of the wall.

        V:
            The voltage of the source.

        R:
            The resistance of the heater.

        C:
            The heat capacity of the room.

        wall_capacity:
            The heat capacity of the wall.

        wall_conductance:
            The thermal conductance between the room and the wall.

        surface_conductance:
            The convective conductance between the wall and the outside air.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.engineering.control import HeatedRoomSystem
        >>> HeatedRoomSystem(setting=22.0).prepare().system_.interactions[-1].threshold
        21.5
    """

    field = 'engineering/control'
    topics = ('control', 'thermal', 'acausal components', 'hysteresis', 'ode', 'events')
    references = ('Fritzson 2015',)
    until = 14400.0
    dt = 10.0
    observables = (Observable('power', _get_heating_power),)

    def __init__(
        self,
        T: float = 10.0,
        *,
        setting: float = 20.0,
        band: float = 1.0,
        outside: float = 5.0,
        V: float = 230.0,
        R: float = 25.0,
        C: float = 2e5,
        wall_capacity: float = 1e6,
        wall_conductance: float = 100.0,
        surface_conductance: float = 50.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.T = T
        self.setting = setting
        self.band = band
        self.outside = outside
        self.V = V
        self.R = R
        self.C = C
        self.wall_capacity = wall_capacity
        self.wall_conductance = wall_conductance
        self.surface_conductance = surface_conductance

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        upper, lower = self.setting + self.band / 2, self.setting - self.band / 2
        self.system_ = System(
            self.label,
            elements=[
                Element('supply', values={'v': 0.0}),
                Element('neutral', values={'v': 0.0}),
                Element('source', values={'i': 0.0}),
                Element('earth', values={'i': 0.0}),
                Element('heater', values={'closed': 1.0}),
                Element('room', values={'T': self.T}),
                Element('wall', values={'T': self.outside}),
            ],
            interactions=[
                VoltageSourceInteraction('mains', a='supply', b='neutral', branch='source', V=self.V),
                GroundInteraction('ground', node='neutral', branch='earth'),
                HeatingResistorInteraction(
                    'heating',
                    a='supply',
                    b='neutral',
                    heated='room',
                    switch='heater',
                    R=self.R,
                ),
                HeatCapacityInteraction('air', node='room', C=self.C),
                ThermalConductorInteraction('inner', a='room', b='wall', G=self.wall_conductance),
                HeatCapacityInteraction('masonry', node='wall', C=self.wall_capacity),
                ConvectionInteraction('surface', node='wall', h=self.surface_conductance, fluid=self.outside),
                SwitchInteraction('open_switch', sensor='room', switch='heater', threshold=upper, value=0.0),
                SwitchInteraction(
                    'close_switch',
                    sensor='room',
                    switch='heater',
                    threshold=lower,
                    value=1.0,
                    direction=-1,
                ),
            ],
        )
