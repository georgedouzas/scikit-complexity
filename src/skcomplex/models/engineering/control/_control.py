"""Control models: feedback loops that keep or drive a system towards a goal."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ....core.interactions import ConnectionInteraction, CrossingInteraction
from ....core.system import System
from ....engine.solvers import RK4Solver
from ....specs import ObservableSpec
from ..._catalog import register_model
from ..common.electrical import GroundElement, VoltageSourceElement
from ._controlled_process import ControlledProcessElement
from ._room import RoomElement
from ._satellite import SatelliteElement
from ._swinging_pendulum import SwingingPendulumElement
from ._thermal import ConvectionElement, HeatCapacityElement, HeatingResistorElement, ThermalConductorElement


def _energy_of_satellite(s: SimpleNamespace) -> Expression:
    """Return the total effective energy of the satellite of the system."""
    satellite = s.satellite
    return satellite.s_d1**2 / 2 + satellite.Lz**2 / (2 * satellite.s**2) - 1 / satellite.s


@register_model(
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
            The model, with the room temperature `room.T` in hours.

    Examples:
        >>> from skcomplex.models.engineering.control import thermostat
        >>> thermostat(leak=0.3).elements[0].leak
        0.3
    """
    room = RoomElement('room', T=T, setting=setting, furnace=furnace, leak=leak, outside=outside, swing=swing)
    return System(elements=[room])


@register_model(
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
            The model, with the output `process.y` and the integral of the error `process.integral`.

    Examples:
        >>> from skcomplex.models.engineering.control import pid_loop
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


@register_model(
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
            The model, with the angle `pendulum.theta`, the position of the pivot `pendulum.f`, and their velocities.

    Examples:
        >>> from skcomplex.models.engineering.control import pendulum_swing_up
        >>> pendulum_swing_up(mu=0.2).elements[0].mu
        0.2
    """
    pendulum = SwingingPendulumElement('pendulum', theta=theta, theta_d1=omega0, f=0.0, f_d1=0.0, omega=omega, mu=mu)
    return System(elements=[pendulum])


@register_model(
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
            The model, with the distance `satellite.s`, the radial velocity `satellite.s_d1`, the angle
            `satellite.theta`, and the observable `energy`.

    Examples:
        >>> from skcomplex.models.engineering.control import satellite_transfer
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


def _heating_power(s: SimpleNamespace) -> Expression:
    """Return the heat that the resistor delivers to the room."""
    heat: Expression = -s.heater.h_Q
    return heat


@register_model(
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
            The model, with the room temperature `room.T` and the switch `heater.on`, in seconds.

    Examples:
        >>> from skcomplex.models.engineering.control import heated_room
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
