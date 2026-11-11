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
from ....core.components.laws import BaseLaw, EquationLaw
from ....core.components.laws.utils import clip, derivative, sign, step, where
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import RK4Solver
from ....core.specs import ConnectorSpec, DeclaredVariableSpec, ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations
from ..common.electrical import ELECTRICAL, GroundElement, TwoPinElement, VoltageSourceElement

DISTURBANCE_TIME = 10.0
CHANGE_TIME = 20.0
HOURS_PER_DAY = 24.0
THERMAL = ConnectorSpec('thermal', potential='T', flow='Q')
POSITIVE = RealSet.open(0, oo)


class HeatCapacityElement(Element):
    """Body of one temperature `T`, with `C T' = a_Q` and `T = a_T`.

    Args:
        label:
            The label of the element, unique among the components of its system.

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

    law_classes = (EquationLaw,)

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
    """Conductor between two ports.

    It has the heat flow `a_Q = G (a_T - b_T)` and `a_Q + b_Q = 0`.

    Args:
        label:
            The label of the element, unique among the components of its system.

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

    law_classes = (EquationLaw,)

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
    """Surface that exchanges heat with a fluid of a fixed temperature.

    It has `a_Q = h (a_T - fluid)`.

    Args:
        label:
            The label of the element, unique among the components of its system.

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

    law_classes = (EquationLaw,)

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
    """Resistor that delivers its electric power to a thermal port `h`.

    It delivers the power while its switch `on` is closed. The current is `on (p_v - n_v) / R`, and the heat flow into
    the thermal port is `-(p_v - n_v) p_i`.

    Args:
        label:
            The label of the element, unique among the components of its system.

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
        >>> ThermostatSystem(leak=0.3).prepare().system_.elements[0].get_value('leak')
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
    ) -> None:
        super().__init__(label=label)
        self.T = T
        self.setting = setting
        self.furnace = furnace
        self.leak = leak
        self.outside = outside
        self.swing = swing

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'room'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'T': DeclaredVariableSpec(self.T, StateSpec()),
            'setting': DeclaredVariableSpec(self.setting, ParameterSpec()),
            'furnace': DeclaredVariableSpec(self.furnace, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'leak': DeclaredVariableSpec(self.leak, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'outside': DeclaredVariableSpec(self.outside, ParameterSpec()),
            'swing': DeclaredVariableSpec(self.swing, ParameterSpec(domain=RealSet.closed_open(0, oo))),
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
        >>> PidLoopSystem(Ki=0.0).prepare().system_.elements[0].get_value('Ki')
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
    ) -> None:
        super().__init__(label=label)
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

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'y': DeclaredVariableSpec(self.y, StateSpec()),
            'integral': DeclaredVariableSpec(0.0, StateSpec()),
            'tau': DeclaredVariableSpec(self.tau, ParameterSpec(domain=RealSet.open(0, oo))),
            'gain': DeclaredVariableSpec(self.gain, ParameterSpec()),
            'Kp': DeclaredVariableSpec(self.Kp, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'Ki': DeclaredVariableSpec(self.Ki, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'Kd': DeclaredVariableSpec(self.Kd, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'setpoint': DeclaredVariableSpec(self.setpoint, ParameterSpec()),
            'change': DeclaredVariableSpec(self.change, ParameterSpec()),
            'disturbance': DeclaredVariableSpec(self.disturbance, ParameterSpec()),
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
        >>> PendulumSwingUpSystem(mu=0.2).prepare().system_.elements[0].get_value('mu')
        0.2
    """

    field = 'engineering/control'
    topics = ('control', 'bang-bang', 'pendulum', 'energy', 'ode')
    references = ('Åström and Furuta 2000', 'Dilão 2023')
    until = 100.0
    dt = 0.1
    solver = RK4Solver(dt=0.01)

    def __init__(
        self,
        theta: float = 0.0,
        omega0: float = 0.5,
        *,
        omega: float = 1.5652475842498528,
        mu: float = 0.1,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.theta = theta
        self.omega0 = omega0
        self.omega = omega
        self.mu = mu

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'pendulum'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'theta': DeclaredVariableSpec(self.theta, StateSpec(order=2, initial=(self.omega0,))),
            'f': DeclaredVariableSpec(0.0, StateSpec(order=2, initial=(0.0,))),
            'omega': DeclaredVariableSpec(self.omega, ParameterSpec(domain=RealSet.open(0, oo))),
            'mu': DeclaredVariableSpec(self.mu, ParameterSpec(domain=RealSet.closed_open(0, oo))),
        }

    def _get_equations(self) -> Equations:
        """Return the controlled rate of the angle and the rate of the control integral."""
        return {
            'theta': lambda s: (
                -(s.target.omega**2) * sin(s.target.theta)
                - PendulumSwingUpSystem._swing_up_control(s) * cos(s.target.theta)
            ),
            'f': PendulumSwingUpSystem._swing_up_control,
        }

    @staticmethod
    def _get_pendulum_energy(s: SimpleNamespace) -> Expression:
        """Return the energy per unit inertia of the pendulum, `θ'^2 / 2 - ω^2 cos θ`."""
        return derivative(s.target.theta, 1) ** 2 / 2 - s.target.omega**2 * cos(s.target.theta)

    @staticmethod
    def _swing_up_control(s: SimpleNamespace) -> Expression:
        """Return the bang-bang control of the energy of the pendulum.

        It is `u = -μ sign(θ' cos θ) sign(E_s - E)`.
        """
        gap = s.target.omega**2 - PendulumSwingUpSystem._get_pendulum_energy(s)
        return -s.target.mu * sign(derivative(s.target.theta, 1) * cos(s.target.theta)) * sign(gap)


def _get_energy_of_satellite(s: SimpleNamespace) -> Expression:
    """Return the total effective energy of the satellite of the system."""
    satellite = s.satellite
    return satellite.s_d1**2 / 2 + satellite.Lz**2 / (2 * satellite.s**2) - 1 / satellite.s


class SatelliteTransferSystem(DynamicalSystem):
    """A satellite moved between Kepler orbits by radial thrust.

    The orbits have equal angular momentum. In normalised polar coordinates the satellite follows
    `s'' = L_z^2 / s^3 - 1 / s^2 - ε sigma` and `θ' = L_z / s^2`. Radial thrust keeps the angular momentum and changes
    the effective energy `H = s'^2 / 2 + L_z^2 / (2 s^2) - 1 / s` at the rate `-ε sigma s'`, so the controller pushes
    inwards, `sigma = 1`, while the satellite falls and outwards, `sigma = -1`, while it rises, until `H` reaches the
    final energy. From the periapsis of the orbit with `H = -0.7`, the transfer to `H = -0.2` takes about 9.81.

    The system has the distance `satellite.s`, the radial velocity `satellite.s_d1`, the angle `satellite.theta`, and
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
        >>> SatelliteTransferSystem().prepare().system_.elements[0].get_value('Lz')
        0.7
    """

    field = 'engineering/control'
    topics = ('control', 'orbits', 'celestial mechanics', 'ode')
    references = ('Dilão et al. 2022', 'Dilão 2023')
    until = 20.0
    dt = 0.05
    solver = RK4Solver(dt=0.005)
    observables = (ObservableSpec('energy', _get_energy_of_satellite),)

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
    ) -> None:
        super().__init__(label=label)
        self.s = s
        self.velocity = velocity
        self.theta = theta
        self.Lz = Lz
        self.epsilon = epsilon
        self.H_final = H_final

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'satellite'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            's': DeclaredVariableSpec(self.s, StateSpec(domain=RealSet.open(0, oo), order=2, initial=(self.velocity,))),
            'theta': DeclaredVariableSpec(self.theta, StateSpec()),
            'Lz': DeclaredVariableSpec(self.Lz, ParameterSpec(domain=RealSet.open(0, oo))),
            'epsilon': DeclaredVariableSpec(self.epsilon, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'H_final': DeclaredVariableSpec(self.H_final, ParameterSpec()),
        }

    def _get_equations(self) -> Equations:
        """Return the radial rate under thrust and the rate of the angle."""
        return {
            's': lambda s: (
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
        return derivative(s.target.s, 1) ** 2 / 2 + s.target.Lz**2 / (2 * s.target.s**2) - 1 / s.target.s

    @staticmethod
    def _get_thrust_direction(s: SimpleNamespace) -> Expression:
        """Return the thrust `sigma`.

        It points inwards while the satellite falls, outwards while it rises, to the end energy.
        """
        return where(
            SatelliteTransferSystem._get_satellite_energy(s) < s.target.H_final,
            where(derivative(s.target.s, 1) < 0, 1, -1),
            0,
        )


def _get_heating_power(s: SimpleNamespace) -> Expression:
    """Return the heat that the resistor delivers to the room."""
    heat: Expression = -s.heater.h_Q
    return heat


class HeatedRoomSystem(BaseSystem):
    """A room assembled from thermal and electrical components.

    It is heated under a thermostat with hysteresis. A voltage source drives a heating resistor whose heat enters the
    room, and the room loses heat through a wall of its own heat capacity to the outside air. The thermostat opens the
    switch of the resistor when the room rises through `setting + band / 2` and closes it when the room falls through
    `setting - band / 2`. The system has the room temperature `room.T` and the switch `heater.on`, in seconds.

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

    Examples:
        >>> from skcomplex.systems.engineering.control import HeatedRoomSystem
        >>> HeatedRoomSystem(setting=22.0).prepare().system_.interactions[-1].direction
        -1
    """

    field = 'engineering/control'
    topics = ('control', 'thermal', 'acausal components', 'hysteresis', 'ode', 'events')
    references = ('Fritzson 2015',)
    until = 14400.0
    dt = 10.0
    observables = (ObservableSpec('power', _get_heating_power),)

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
    ) -> None:
        super().__init__(label)
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

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        upper, lower = self.setting + self.band / 2, self.setting - self.band / 2
        return CompositeSystem(
            self.label,
            elements=[
                VoltageSourceElement('source', V=self.V),
                HeatingResistorElement('heater', R=self.R),
                GroundElement('ground'),
                HeatCapacityElement('room', T=self.T, C=self.C),
                ThermalConductorElement('inner', G=self.wall_conductance),
                HeatCapacityElement('wall', T=self.outside, C=self.wall_capacity),
                ConvectionElement('surface', h=self.surface_conductance, fluid=self.outside),
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
