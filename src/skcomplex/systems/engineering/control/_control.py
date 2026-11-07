"""Control systems: feedback loops that keep or drive a system towards a goal."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import Expression, cos, oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import ConnectionInteraction, CrossingInteraction
from ....core.components.laws import BaseLaw, EquationLaw, RateLaw
from ....core.components.laws.utils import clip, derivative, sign, step, where
from ....core.components.system import System
from ....core.engine.solvers import RK4Solver
from ....core.specs import ConnectorSpec, ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ..common.electrical import ELECTRICAL, GroundElement, TwoPinElement, VoltageSourceElement

DISTURBANCE_TIME = 10.0


CHANGE_TIME = 20.0


class ControlledProcessElement(Element):
    """First-order process under a PID controller whose derivative term is solved into the time constant."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y: Annotated[object, StateSpec()] = 0.0,
        integral: Annotated[object, StateSpec()] = 0.0,
        tau: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        gain: Annotated[object, ParameterSpec()] = 1.0,
        Kp: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        Ki: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        Kd: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        setpoint: Annotated[object, ParameterSpec()] = 1.0,
        change: Annotated[object, ParameterSpec()] = 0.0,
        disturbance: Annotated[object, ParameterSpec()] = -0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y = y
        self.integral = integral
        self.tau = tau
        self.gain = gain
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.change = change
        self.disturbance = disturbance

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the output `y` and of the integral of the error."""
        return [
            RateLaw(
                s.y,
                (
                    -s.y
                    + s.gain * (s.Kp * ControlledProcessElement._error(s) + s.Ki * s.integral)
                    + s.disturbance * step(s.t, DISTURBANCE_TIME)
                )
                / (s.tau + s.gain * s.Kd),
            ),
            RateLaw(s.integral, ControlledProcessElement._error(s)),
        ]

    @staticmethod
    def _error(s: SimpleNamespace) -> Expression:
        """Return the error of the controlled process, the set point after its change less the output."""
        return s.setpoint + s.change * step(s.t, CHANGE_TIME) - s.y


HOURS_PER_DAY = 24.0


class RoomElement(Element):
    """Room heated by a furnace towards a setting and leaking heat to an outside temperature with a daily swing."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        T: Annotated[object, StateSpec()] = 10.0,
        setting: Annotated[object, ParameterSpec()] = 18.0,
        furnace: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        leak: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        outside: Annotated[object, ParameterSpec()] = 10.0,
        swing: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.T = T
        self.setting = setting
        self.furnace = furnace
        self.leak = leak
        self.outside = outside
        self.swing = swing

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the saturated heating and the leak of the temperature `T`."""
        return [
            RateLaw(
                s.T,
                clip(s.setting - s.T, 0, s.furnace)
                - s.leak * (s.T - s.outside - s.swing * cos(2 * pi * (s.t - HOURS_PER_DAY / 2) / HOURS_PER_DAY)),
            ),
        ]


class SatelliteElement(Element):
    """Satellite on a planar Kepler orbit whose radial thrust moves it to an orbit of a final energy."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        s: Annotated[object, StateSpec(domain=RealSet.open(0, oo), order=2)] = 0.314,
        s_d1: float = 0.0,
        theta: Annotated[object, StateSpec()] = 0.0,
        Lz: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.7,
        epsilon: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        H_final: Annotated[object, ParameterSpec()] = -0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.s = s
        self.s_d1 = s_d1
        self.theta = theta
        self.Lz = Lz
        self.epsilon = epsilon
        self.H_final = H_final

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the radial rate under thrust and the rate of the angle."""
        return [
            RateLaw(s.s, s.Lz**2 / s.s**3 - 1 / s.s**2 - s.epsilon * SatelliteElement._thrust_direction(s)),
            RateLaw(s.theta, s.Lz / s.s**2),
        ]

    @staticmethod
    def _satellite_energy(s: SimpleNamespace) -> Expression:
        """Return the total effective energy of the satellite, `s'^2 / 2 + L_z^2 / (2 s^2) - 1 / s`."""
        return derivative(s.s, 1) ** 2 / 2 + s.Lz**2 / (2 * s.s**2) - 1 / s.s

    @staticmethod
    def _thrust_direction(s: SimpleNamespace) -> Expression:
        """Return the thrust `sigma`: inwards while the satellite falls, outwards while it rises, to the end energy."""
        return where(SatelliteElement._satellite_energy(s) < s.H_final, where(derivative(s.s, 1) < 0, 1, -1), 0)


class SwingingPendulumElement(Element):
    """Pendulum swung up by a bang-bang control of its energy, with `f` integrating the control."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec(order=2)] = 0.0,
        theta_d1: float = 0.0,
        f: Annotated[object, StateSpec(order=2)] = 0.0,
        f_d1: float = 0.0,
        omega: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.5652475842498528,
        mu: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.theta_d1 = theta_d1
        self.f = f
        self.f_d1 = f_d1
        self.omega = omega
        self.mu = mu

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the controlled rate of the angle and the rate of the control integral."""
        return [
            RateLaw(
                s.theta,
                -(s.omega**2) * sin(s.theta) - SwingingPendulumElement._swing_up_control(s) * cos(s.theta),
            ),
            RateLaw(s.f, SwingingPendulumElement._swing_up_control(s)),
        ]

    @staticmethod
    def _pendulum_energy(s: SimpleNamespace) -> Expression:
        """Return the energy per unit inertia of the pendulum, `θ'^2 / 2 - ω^2 cos θ`."""
        return derivative(s.theta, 1) ** 2 / 2 - s.omega**2 * cos(s.theta)

    @staticmethod
    def _swing_up_control(s: SimpleNamespace) -> Expression:
        """Return the bang-bang control `u = -μ sign(θ' cos θ) sign(E_s - E)` of the energy of the pendulum."""
        gap = s.omega**2 - SwingingPendulumElement._pendulum_energy(s)
        return -s.mu * sign(derivative(s.theta, 1) * cos(s.theta)) * sign(gap)


THERMAL = ConnectorSpec('thermal', potential='T', flow='Q')
POSITIVE = RealSet.open(0, oo)


class HeatCapacityElement(Element):
    """Body of one temperature `T`, with `C T' = a_Q` and `T = a_T`.

    Args:
        label:
            The label of the element, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        a_T:
            The temperature of the port.

        a_Q:
            The heat flow into the port.

        T:
            The initial temperature of the body.

        C:
            The heat capacity.

    Examples:
        >>> from skcomplex.systems.engineering.control import HeatCapacityElement
        >>> HeatCapacityElement('room', T=15.0).T
        15.0
    """

    ports: ClassVar[Mapping[str, ConnectorSpec]] = {'a': THERMAL}

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a_T: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        a_Q: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        T: Annotated[object, StateSpec()] = 20.0,
        C: Annotated[object, ParameterSpec(domain=POSITIVE)] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a_T = a_T
        self.a_Q = a_Q
        self.T = T
        self.C = C

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the heat balance of the body and the temperature of its port."""
        return [EquationLaw(s.C * derivative(s.T, 1), s.a_Q), EquationLaw(s.T, s.a_T)]


class ThermalConductorElement(Element):
    """Conductor between two ports, with the heat flow `a_Q = G (a_T - b_T)` and `a_Q + b_Q = 0`.

    Args:
        label:
            The label of the element, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        a_T:
            The temperature of the first port.

        a_Q:
            The heat flow into the first port.

        b_T:
            The temperature of the second port.

        b_Q:
            The heat flow into the second port.

        G:
            The thermal conductance.

    Examples:
        >>> from skcomplex.systems.engineering.control import ThermalConductorElement
        >>> ThermalConductorElement('wall', G=2.0).G
        2.0
    """

    ports: ClassVar[Mapping[str, ConnectorSpec]] = {'a': THERMAL, 'b': THERMAL}

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a_T: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        a_Q: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        b_T: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        b_Q: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        G: Annotated[object, ParameterSpec(domain=POSITIVE)] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a_T = a_T
        self.a_Q = a_Q
        self.b_T = b_T
        self.b_Q = b_Q
        self.G = G

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the heat flow through the conductor and its balance."""
        return [EquationLaw(s.a_Q, s.G * (s.a_T - s.b_T)), EquationLaw(s.a_Q + s.b_Q, 0)]


class ConvectionElement(Element):
    """Surface that exchanges heat with a fluid of a fixed temperature, with `a_Q = h (a_T - fluid)`.

    Args:
        label:
            The label of the element, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        a_T:
            The temperature of the port.

        a_Q:
            The heat flow into the port.

        h:
            The convective conductance of the surface.

        fluid:
            The temperature of the fluid.

    Examples:
        >>> from skcomplex.systems.engineering.control import ConvectionElement
        >>> ConvectionElement('outside', fluid=5.0).fluid
        5.0
    """

    ports: ClassVar[Mapping[str, ConnectorSpec]] = {'a': THERMAL}

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a_T: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        a_Q: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        h: Annotated[object, ParameterSpec(domain=POSITIVE)] = 1.0,
        fluid: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a_T = a_T
        self.a_Q = a_Q
        self.h = h
        self.fluid = fluid

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the heat flow into the fluid."""
        return [EquationLaw(s.a_Q, s.h * (s.a_T - s.fluid))]


class HeatingResistorElement(TwoPinElement):
    """Resistor that delivers its electric power to a thermal port `h` while its switch `on` is closed.

    The current is `on (p_v - n_v) / R`, and the heat flow into the thermal port is `-(p_v - n_v) p_i`.

    Args:
        label:
            The label of the element, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        p_v:
            The potential of the positive pin.

        p_i:
            The current into the positive pin.

        n_v:
            The potential of the negative pin.

        n_i:
            The current into the negative pin.

        h_T:
            The temperature of the thermal port.

        h_Q:
            The heat flow into the thermal port.

        on:
            The initial position of the switch, 1 for closed and 0 for open.

        R:
            The resistance.

    Examples:
        >>> from skcomplex.systems.engineering.control import HeatingResistorElement
        >>> sorted(HeatingResistorElement.ports)
        ['h', 'n', 'p']
    """

    ports: ClassVar[Mapping[str, ConnectorSpec]] = {'p': ELECTRICAL, 'n': ELECTRICAL, 'h': THERMAL}

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p_v: object = 0.0,
        p_i: object = 0.0,
        n_v: object = 0.0,
        n_i: object = 0.0,
        h_T: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        h_Q: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        on: Annotated[object, StateSpec()] = 1.0,
        R: Annotated[object, ParameterSpec(domain=POSITIVE)] = 1.0,
    ) -> None:
        super().__init__(label, visual, p_v=p_v, p_i=p_i, n_v=n_v, n_i=n_i)
        self.h_T = h_T
        self.h_Q = h_Q
        self.on = on
        self.R = R

    def _get_branch_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the switched current of the resistor and the heat it delivers.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The equation laws of the branch.
        """
        voltage = s.p_v - s.n_v
        return [EquationLaw(s.R * s.p_i, s.on * voltage), EquationLaw(s.h_Q, -voltage * s.p_i)]


def _energy_of_satellite(s: SimpleNamespace) -> Expression:
    """Return the total effective energy of the satellite of the system."""
    satellite = s.satellite
    return satellite.s_d1**2 / 2 + satellite.Lz**2 / (2 * satellite.s**2) - 1 / satellite.s


def _heating_power(s: SimpleNamespace) -> Expression:
    """Return the heat that the resistor delivers to the room."""
    heat: Expression = -s.heater.h_Q
    return heat


@register_system(
    field='engineering/control',
    tags=('control', 'feedback', 'stocks and flows', 'ode'),
    references=('Meadows 2008',),
    defaults={'until': 24.0, 'dt': 0.25},
)
def thermostat(
    T: float = 10.0,
    *,
    setting: float = 18.0,
    furnace: float = 5.0,
    leak: float = 0.1,
    outside: float = 10.0,
    swing: float = 0.0,
) -> System:
    """Return a room heated by a furnace under a thermostat while heat leaks to the outside.

    The furnace adds the gap between the setting and the room temperature, at most `furnace` degrees per hour, and
    nothing when the room is warmer than the setting. The room loses `leak` of its difference to the outside
    temperature per hour, so it settles below the setting, at `(setting + leak * outside) / (1 + leak)` for a
    constant outside. The outside temperature swings by `swing` around `outside` over a day, warmest at noon.

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

    Returns:
        system:
            The system, with the room temperature `room.T` in hours.

    Examples:
        >>> from skcomplex.systems.engineering.control import thermostat
        >>> thermostat(leak=0.3).elements[0].leak
        0.3
    """
    room = RoomElement('room', T=T, setting=setting, furnace=furnace, leak=leak, outside=outside, swing=swing)
    return System(elements=[room])


@register_system(
    field='engineering/control',
    tags=('control', 'feedback', 'pid', 'governance', 'ode'),
    references=('Mobus 2022', 'Åström and Murray 2008'),
    defaults={'until': 40.0, 'dt': 0.1},
)
def pid_loop(
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
) -> System:
    """Return a work process whose output a PID controller keeps at a set point against a disturbance.

    The output follows the first-order process `τ y' = -y + gain u + d(t)`, and the controller sets
    `u = Kp e + Ki ∫ e dt - Kd y'` from the error `e = r(t) - y`, with the derivative taken on the output, so
    `(τ + gain Kd) y' = -y + gain (Kp e + Ki ∫ e dt) + d(t)`. The disturbance `d` steps in at time 10, and a
    coordinator moves the set point `r` by `change` at time 20. The integral action removes the error that the
    disturbance leaves under proportional control alone.

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

    Returns:
        system:
            The system, with the output `process.y` and the integral of the error `process.integral`.

    Examples:
        >>> from skcomplex.systems.engineering.control import pid_loop
        >>> pid_loop(Ki=0.0).elements[0].Ki
        0.0
    """
    process = ControlledProcessElement(
        'process',
        y=y,
        integral=0.0,
        tau=tau,
        gain=gain,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        setpoint=setpoint,
        change=change,
        disturbance=disturbance,
    )
    return System(elements=[process])


@register_system(
    field='engineering/control',
    tags=('control', 'bang-bang', 'pendulum', 'energy', 'ode'),
    references=('Åström and Furuta 2000', 'Dilão 2023'),
    defaults={'until': 100.0, 'dt': 0.1, 'solver': RK4Solver(dt=0.01)},
)
def pendulum_swing_up(
    theta: float = 0.0,
    omega0: float = 0.5,
    *,
    omega: float = 1.5652475842498528,
    mu: float = 0.1,
) -> System:
    """Return a pendulum swung up to its inverted position by a bang-bang acceleration of its pivot.

    The pivot of an acrobat's pendulum moves horizontally with acceleration `f'' = u`, so the pendulum follows
    `θ'' + ω^2 sin θ + u cos θ = 0`. The control `u = -μ sign(θ' cos θ) sign(E_s - E)` pumps the energy
    `E = θ'^2 / 2 - ω^2 cos θ` towards `E_s = ω^2`, the energy of the inverted position, from below or from above.

    Args:
        theta:
            The initial angle from the hanging position.

        omega0:
            The initial angular velocity.

        omega:
            The natural frequency `sqrt(g / l)`, by default for `g = 9.8` and `l = 4`.

        mu:
            The size of the acceleration of the pivot.

    Returns:
        system:
            The system, with the angle `pendulum.theta`, the position of the pivot `pendulum.f`, and their velocities.

    Examples:
        >>> from skcomplex.systems.engineering.control import pendulum_swing_up
        >>> pendulum_swing_up(mu=0.2).elements[0].mu
        0.2
    """
    pendulum = SwingingPendulumElement('pendulum', theta=theta, theta_d1=omega0, f=0.0, f_d1=0.0, omega=omega, mu=mu)
    return System(elements=[pendulum])


@register_system(
    field='engineering/control',
    tags=('control', 'orbits', 'celestial mechanics', 'ode'),
    references=('Dilão et al. 2022', 'Dilão 2023'),
    defaults={'until': 20.0, 'dt': 0.05, 'solver': RK4Solver(dt=0.005)},
    observables=lambda system: [ObservableSpec('energy', _energy_of_satellite)],
)
def satellite_transfer(
    s: float = 0.314,
    velocity: float = 0.0,
    theta: float = 0.0,
    *,
    Lz: float = 0.7,
    epsilon: float = 0.1,
    H_final: float = -0.2,
) -> System:
    """Return a satellite moved between Kepler orbits of equal angular momentum by radial thrust.

    In normalised polar coordinates the satellite follows `s'' = L_z^2 / s^3 - 1 / s^2 - ε sigma` and
    `θ' = L_z / s^2`. Radial thrust keeps the angular momentum and changes the effective energy
    `H = s'^2 / 2 + L_z^2 / (2 s^2) - 1 / s` at the rate `-ε sigma s'`, so the controller pushes inwards, `sigma = 1`,
    while the satellite falls and outwards, `sigma = -1`, while it rises, until `H` reaches the final energy. From the
    periapsis of the orbit with `H = -0.7`, the transfer to `H = -0.2` takes about 9.81.

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

    Returns:
        system:
            The system, with the distance `satellite.s`, the radial velocity `satellite.s_d1`, the angle
            `satellite.theta`, and the observable `energy`.

    Examples:
        >>> from skcomplex.systems.engineering.control import satellite_transfer
        >>> satellite_transfer().elements[0].Lz
        0.7
    """
    satellite = SatelliteElement(
        'satellite',
        s=s,
        s_d1=velocity,
        theta=theta,
        Lz=Lz,
        epsilon=epsilon,
        H_final=H_final,
    )
    return System(elements=[satellite])


@register_system(
    field='engineering/control',
    tags=('control', 'thermal', 'acausal components', 'hysteresis', 'ode', 'events'),
    references=('Fritzson 2015',),
    defaults={'until': 14400.0, 'dt': 10.0},
    observables=lambda system: [ObservableSpec('power', _heating_power)],
)
def heated_room(
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
) -> System:
    """Return a room assembled from thermal and electrical components, heated under a thermostat with hysteresis.

    A voltage source drives a heating resistor whose heat enters the room, and the room loses heat through a wall of its
    own heat capacity to the outside air. The thermostat opens the switch of the resistor when the room rises through
    `setting + band / 2` and closes it when the room falls through `setting - band / 2`.

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

    Returns:
        system:
            The system, with the room temperature `room.T` and the switch `heater.on`, in seconds.

    Examples:
        >>> from skcomplex.systems.engineering.control import heated_room
        >>> heated_room(setting=22.0).interactions[-1].direction
        -1
    """
    upper, lower = setting + band / 2, setting - band / 2
    return System(
        elements=[
            VoltageSourceElement('source', V=V),
            HeatingResistorElement('heater', R=R),
            GroundElement('ground'),
            HeatCapacityElement('room', T=T, C=C),
            ThermalConductorElement('inner', G=wall_conductance),
            HeatCapacityElement('wall', T=outside, C=wall_capacity),
            ConvectionElement('surface', h=surface_conductance, fluid=outside),
        ],
        interactions=[
            ConnectionInteraction('supply', members=['source', 'heater'], ports=['p', 'p']),
            ConnectionInteraction('neutral', members=['source', 'heater', 'ground'], ports=['n', 'n', 'p']),
            ConnectionInteraction('air', members=['heater', 'room', 'inner'], ports=['h', 'a', 'a']),
            ConnectionInteraction('masonry', members=['inner', 'wall', 'surface'], ports=['b', 'a', 'a']),
            CrossingInteraction(
                'open_switch',
                target='heater',
                guard=lambda s: s.target.h_T - upper,
                reset={'on': lambda s: 0.0},
                direction=1,
            ),
            CrossingInteraction(
                'close_switch',
                target='heater',
                guard=lambda s: s.target.h_T - lower,
                reset={'on': lambda s: 1.0},
                direction=-1,
            ),
        ],
    )
