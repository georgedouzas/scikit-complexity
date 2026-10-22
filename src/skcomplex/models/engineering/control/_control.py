"""Control models: feedback loops that keep or drive a system towards a goal."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import cos, pi, sin

from ....core.base import Domain, Expression, parameter, state
from ....core.model import Element, Observable, Rate, Symbols, System, derivative, step, where
from ....core.runtime import register_model
from ....core.solvers import RK4

HOURS_PER_DAY = 24.0
DISTURBANCE_TIME = 10.0
CHANGE_TIME = 20.0


def _sign(value: Expression) -> Expression:
    """Return the sign of an expression, 0 at 0."""
    return where(value > 0, 1, where(value < 0, -1, 0))


def _error(s: Symbols) -> Expression:
    """Return the error of the controlled process, the set point after its change less the output."""
    return s.setpoint + s.change * step(s.t, CHANGE_TIME) - s.y


def _pendulum_energy(s: Symbols) -> Expression:
    """Return the energy per unit inertia of the pendulum, `θ'^2 / 2 - ω^2 cos θ`."""
    return derivative(s.theta, 1) ** 2 / 2 - s.omega**2 * cos(s.theta)


def _swing_up_control(s: Symbols) -> Expression:
    """Return the bang-bang control `u = -μ sign(θ' cos θ) sign(E_s - E)` of the energy of the pendulum."""
    gap = s.omega**2 - _pendulum_energy(s)
    return -s.mu * _sign(derivative(s.theta, 1) * cos(s.theta)) * _sign(gap)


def _satellite_energy(s: Symbols) -> Expression:
    """Return the total effective energy of the satellite, `s'^2 / 2 + L_z^2 / (2 s^2) - 1 / s`."""
    return derivative(s.s, 1) ** 2 / 2 + s.Lz**2 / (2 * s.s**2) - 1 / s.s


def _thrust_direction(s: Symbols) -> Expression:
    """Return the thrust `sigma`: inwards while the satellite falls, outwards while it rises, until the final energy."""
    return where(_satellite_energy(s) < s.H_final, where(derivative(s.s, 1) < 0, 1, -1), 0)


def _energy_of_satellite(s: Symbols) -> Expression:
    """Return the total effective energy of the satellite of the system."""
    satellite = s.satellite
    return satellite.s_d1**2 / 2 + satellite.Lz**2 / (2 * satellite.s**2) - 1 / satellite.s


Room = Element.define(
    'Room',
    states={'T': 10.0},
    parameters={
        'setting': 18.0,
        'furnace': parameter(default=5.0, domain=Domain.NONNEGATIVE),
        'leak': parameter(default=0.1, domain=Domain.NONNEGATIVE),
        'outside': 10.0,
        'swing': parameter(default=0.0, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [
        Rate(
            s.T,
            where(s.setting - s.T > s.furnace, s.furnace, where(s.setting > s.T, s.setting - s.T, 0))
            - s.leak * (s.T - s.outside - s.swing * cos(2 * pi * (s.t - HOURS_PER_DAY / 2) / HOURS_PER_DAY)),
        ),
    ],
)

ControlledProcess = Element.define(
    'ControlledProcess',
    states={'y': 0.0, 'integral': 0.0},
    parameters={
        'tau': parameter(default=1.0, domain=Domain.POSITIVE),
        'gain': parameter(default=1.0),
        'Kp': parameter(default=2.0, domain=Domain.NONNEGATIVE),
        'Ki': parameter(default=1.0, domain=Domain.NONNEGATIVE),
        'Kd': parameter(default=0.1, domain=Domain.NONNEGATIVE),
        'setpoint': 1.0,
        'change': 0.0,
        'disturbance': -0.5,
    },
    laws=lambda s: [
        Rate(
            s.y,
            (-s.y + s.gain * (s.Kp * _error(s) + s.Ki * s.integral) + s.disturbance * step(s.t, DISTURBANCE_TIME))
            / (s.tau + s.gain * s.Kd),
        ),
        Rate(s.integral, _error(s)),
    ],
)

SwingingPendulum = Element.define(
    'SwingingPendulum',
    states={'theta': state(default=0.0, order=2), 'f': state(default=0.0, order=2)},
    parameters={
        'omega': parameter(default=1.5652475842498528, domain=Domain.POSITIVE),
        'mu': parameter(default=0.1, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [
        Rate(s.theta, -(s.omega**2) * sin(s.theta) - _swing_up_control(s) * cos(s.theta)),
        Rate(s.f, _swing_up_control(s)),
    ],
)

Satellite = Element.define(
    'Satellite',
    states={'s': state(default=0.314, domain=Domain.POSITIVE, order=2), 'theta': 0.0},
    parameters={
        'Lz': parameter(default=0.7, domain=Domain.POSITIVE),
        'epsilon': parameter(default=0.1, domain=Domain.NONNEGATIVE),
        'H_final': -0.2,
    },
    laws=lambda s: [
        Rate(s.s, s.Lz**2 / s.s**3 - 1 / s.s**2 - s.epsilon * _thrust_direction(s)),
        Rate(s.theta, s.Lz / s.s**2),
    ],
)


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
    room = Room('room', T=T, setting=setting, furnace=furnace, leak=leak, outside=outside, swing=swing)
    return System(components=[room])


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
    process = ControlledProcess(
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
    return System(components=[process])


@register_model(
    field='engineering/control',
    tags=('control', 'bang-bang', 'pendulum', 'energy', 'ode'),
    references=('Åström and Furuta 2000', 'Dilão 2023'),
    defaults={'until': 100.0, 'dt': 0.1, 'solver': RK4(dt=0.01)},
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
    pendulum = SwingingPendulum('pendulum', theta=theta, theta_d1=omega0, f=0.0, f_d1=0.0, omega=omega, mu=mu)
    return System(components=[pendulum])


@register_model(
    field='engineering/control',
    tags=('control', 'orbits', 'celestial mechanics', 'ode'),
    references=('Dilão et al. 2022', 'Dilão 2023'),
    defaults={'until': 20.0, 'dt': 0.05, 'solver': RK4(dt=0.005)},
    observables=lambda system: [Observable('energy', _energy_of_satellite)],
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
    satellite = Satellite('satellite', s=s, s_d1=velocity, theta=theta, Lz=Lz, epsilon=epsilon, H_final=H_final)
    return System(components=[satellite])
