"""Electromechanical energy systems: braking energy recovery, homopolar chargers, and loaded rails."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, Expression, parameter, state
from ....core.model import Element, Rate, Symbols, System, derivative, pulse_train, where
from ....core.runtime import register_model


def _dry_friction(speed: Expression, coefficient: Expression) -> Expression:
    """Return dry friction of a coefficient against the direction of a speed, 0 at rest."""
    return where(speed > 0, coefficient, where(speed < 0, -coefficient, 0))


def _wheel_load(s: Symbols) -> Expression:
    """Return the force of the two wheels of a bogie: two pulses per period, `spacing` apart."""
    return pulse_train(s.t, s.T, s.width, s.F) + pulse_train(s.t, s.T, s.width, s.F, start=s.spacing)


RegenerativeCar = Element.define(
    'RegenerativeCar',
    states={'v': state(default=20.0), 'I': 0.0, 'Vs': 0.0},
    parameters={
        'm': parameter(default=1000.0, domain=Domain.POSITIVE),
        'A': parameter(default=1.2, domain=Domain.NONNEGATIVE),
        'L': parameter(default=0.01, domain=Domain.POSITIVE),
        'R': parameter(default=0.05, domain=Domain.POSITIVE),
        'C': parameter(default=680.0, domain=Domain.POSITIVE),
        'alpha': parameter(default=50.0, domain=Domain.NONNEGATIVE),
        'beta': parameter(default=5.0, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [
        Rate(s.v, -(s.A * s.I + _dry_friction(s.v, s.alpha) + s.beta * s.v) / s.m),
        Rate(s.I, (s.A * s.v - s.R * s.I - s.Vs) / s.L),
        Rate(s.Vs, s.I / s.C),
    ],
)

FaradayDisk = Element.define(
    'FaradayDisk',
    states={'I': 0.0, 'U': 0.0, 'omega': 0.0},
    parameters={
        'B': parameter(default=1.5, domain=Domain.NONNEGATIVE),
        'r1': parameter(default=0.025, domain=Domain.POSITIVE),
        'r2': parameter(default=0.55, domain=Domain.POSITIVE),
        'L': parameter(default=0.2, domain=Domain.POSITIVE),
        'C': parameter(default=0.75, domain=Domain.POSITIVE),
        'R1': parameter(default=2.3, domain=Domain.POSITIVE),
        'R2': parameter(default=109.0, domain=Domain.POSITIVE),
        'alpha': parameter(default=0.5, domain=Domain.NONNEGATIVE),
        'beta': parameter(default=0.5, domain=Domain.NONNEGATIVE),
        'J': parameter(default=0.5, domain=Domain.POSITIVE),
        'mu': parameter(default=1.2566370614359173e-4, domain=Domain.NONNEGATIVE),
        'N': parameter(default=250.0, domain=Domain.NONNEGATIVE),
        'l': parameter(default=0.05, domain=Domain.POSITIVE),
        'M': parameter(default=4.8),
        'a0': 8.0,
        'a1': parameter(default=0.75, domain=Domain.NONNEGATIVE),
        'a3': parameter(default=0.5, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [
        Rate(
            s.I,
            (
                -s.R1 * s.I
                - s.U
                + (s.B + s.mu * s.N * s.I / s.l) * (s.r2**2 - s.r1**2) * s.omega / 2
                - (s.a0 + s.a1 * s.I + s.a3 * s.I**3)
            )
            / s.L,
        ),
        Rate(s.U, (s.I - s.U / s.R2) / s.C),
        Rate(
            s.omega,
            (
                s.M
                - _dry_friction(s.omega, s.alpha)
                - s.beta * s.r2 * s.omega
                - s.I * (s.B + s.mu * s.N * s.I / s.l) * (s.r2**2 - s.r1**2) / 2
            )
            / s.J,
        ),
    ],
)

RailSegment = Element.define(
    'RailSegment',
    states={'z': state(default=0.0, order=2), 'zs': state(default=0.0, order=2)},
    parameters={
        'Mr': parameter(default=60.0, domain=Domain.POSITIVE),
        'Ms': parameter(default=156.0, domain=Domain.POSITIVE),
        'kr': parameter(default=6000.0, domain=Domain.NONNEGATIVE),
        'br': parameter(default=300.0, domain=Domain.NONNEGATIVE),
        'ks': parameter(default=1500.0, domain=Domain.NONNEGATIVE),
        'bs': parameter(default=520.0, domain=Domain.NONNEGATIVE),
        'k2': parameter(default=0.0, domain=Domain.NONNEGATIVE),
        'alpha': 1.75,
        'beta': -1.0,
        'F': parameter(default=1000.0, domain=Domain.NONNEGATIVE),
        'T': parameter(default=10.0, domain=Domain.POSITIVE),
        'width': parameter(default=1.0, domain=Domain.POSITIVE),
        'spacing': parameter(default=3.0, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [
        Rate(
            s.z,
            (_wheel_load(s) - s.br * (derivative(s.z, 1) - derivative(s.zs, 1)) - s.kr * (s.z - s.zs)) / s.Mr,
        ),
        Rate(
            s.zs,
            (
                s.br * (derivative(s.z, 1) - derivative(s.zs, 1))
                + s.kr * (s.z - s.zs)
                - s.bs * derivative(s.zs, 1)
                - s.ks * s.zs
                - s.k2 * (s.alpha * s.zs**2 + s.beta * s.zs**3)
            )
            / s.Ms,
        ),
    ],
)


@register_model(
    field='engineering/power_systems',
    tags=('energy', 'electromechanics', 'vehicles', 'ode'),
    references=('Trzaska 2023',),
    defaults={'until': 60.0, 'dt': 0.5},
)
def regenerative_braking(
    v: float = 20.0,
    *,
    m: float = 1000.0,
    A: float = 1.2,
    L: float = 0.01,
    R: float = 0.05,
    C: float = 680.0,
    alpha: float = 50.0,
    beta: float = 5.0,
) -> System:
    """Return a car braked by a Faraday disk generator on its wheel, which charges a supercapacitor.

    The disk generates the voltage `A v`, with the structural constant `A = |B| R^2 / (2 r_w)` of the disk and the
    wheel, and drives the current through an inductor and a resistor into a supercapacitor,
    `L I' = A v - R I - V_s` and `C V_s' = I`. The current brakes the car with the force `A I`, together with dry and
    viscous friction, `m v' = -A I - alpha sign(v) - β v`, so the kinetic energy of the car ends up in the capacitor
    and in the losses.

    Args:
        v:
            The initial speed of the car.

        m:
            The mass of the car.

        A:
            The structural constant of the generator, the voltage per unit speed.

        L:
            The inductance of the charging circuit.

        R:
            The resistance of the charging circuit.

        C:
            The capacitance of the supercapacitor.

        alpha:
            The dry friction force.

        beta:
            The viscous friction coefficient.

    Returns:
        system:
            The model, with the speed `car.v`, the current `car.I`, and the capacitor voltage `car.Vs`.

    Examples:
        >>> from skcomplex.models.engineering.power_systems import regenerative_braking
        >>> regenerative_braking(C=100.0).elements[0].C
        100.0
    """
    car = RegenerativeCar('car', v=v, I=0.0, Vs=0.0, m=m, A=A, L=L, R=R, C=C, alpha=alpha, beta=beta)
    return System(components=[car])


@register_model(
    field='engineering/power_systems',
    tags=('energy', 'electromechanics', 'batteries', 'ode'),
    references=('Trzaska 2023',),
    defaults={'until': 50.0, 'dt': 0.1},
)
def faraday_disk(
    I: float = 0.0,
    U: float = 0.0,
    omega: float = 0.0,
    *,
    B: float = 1.5,
    M: float = 4.8,
    N: float = 250.0,
    R1: float = 2.3,
    R2: float = 109.0,
    L: float = 0.2,
    C: float = 0.75,
) -> System:
    """Return a battery charger driven by a Faraday disk generator whose own current strengthens its field.

    A motor turns the conducting disk of radii `r1` and `r2` with a constant torque `M` in the field `B` of a magnet
    plus the field `μ N I / l` of a coil that the generated current flows through. With `a = B (r2^2 - r1^2) / 2` and
    `b = μ N (r2^2 - r1^2) / (2 l)`, the current, the polarization voltage of the battery, and the angular speed follow
    `L I' = -R1 I - U + (a + b I) ω - E(I)`, `C U' = I - U / R2`, and
    `J ω' = M - alpha sign(ω) - β r2 ω - (a + b I) I`, where the battery voltage is `E(I) = a0 + a1 I + a3 I^3` and the
    last term is the braking torque of the Lorentz force.

    The generator charges the battery only when its voltage `a ω` exceeds the open-circuit voltage `a0 = 8` of the
    battery, at `ω` above 35 for the default field. Below that speed the steady current is a small negative current,
    the battery discharges through the disk and helps the motor, and the steady speed grows linearly with the torque,
    as `(M - alpha) / (β r2)` up to the small Lorentz torque: with the defaults the current settles at -0.040 and the
    speed at 15.7. The coil acts as a negative resistance `b ω` against `R1 + a1`, so above `ω = (R1 + a1) / b`, about
    32 or a torque near 9.5 for the defaults, the steady state loses its stability and the current oscillates around
    zero, swinging between charging and discharging, while the mean braking torque `b <I²>` bends the speed below the
    linear trend. A stronger magnet charges at the default torque: `B = 4` gives a steady current of 0.013.

    Args:
        I:
            The initial charging current.

        U:
            The initial polarization voltage of the battery.

        omega:
            The initial angular speed of the disk.

        B:
            The field of the permanent magnet.

        M:
            The torque of the drive motor.

        N:
            The number of turns of the feedback coil.

        R1:
            The series resistance of the coil, the disk, and the battery.

        R2:
            The polarization resistance of the battery.

        L:
            The inductance of the coil.

        C:
            The polarization capacitance of the battery.

    Returns:
        system:
            The model, with the current `charger.I`, the polarization voltage `charger.U`, and the angular speed
            `charger.omega`.

    Examples:
        >>> from skcomplex.models.engineering.power_systems import faraday_disk
        >>> faraday_disk(M=6.0).elements[0].M
        6.0
    """
    charger = FaradayDisk('charger', I=I, U=U, omega=omega, B=B, M=M, N=N, R1=R1, R2=R2, L=L, C=C)
    return System(components=[charger])


@register_model(
    field='engineering/power_systems',
    tags=('energy', 'vibration', 'railways', 'pulses', 'ode'),
    references=('Trzaska 2023',),
    defaults={'until': 60.0, 'dt': 0.05},
)
def wheel_rail_contact(
    *,
    F: float = 1000.0,
    T: float = 10.0,
    width: float = 1.0,
    spacing: float = 3.0,
    k2: float = 0.0,
    Mr: float = 60.0,
    Ms: float = 156.0,
) -> System:
    """Return a rail segment on a sleeper, loaded by the wheels of passing bogies.

    The rail and the sleeper are lumped masses joined by the pad, a spring and a damper, and the sleeper rests on the
    ballast, `Mr z'' + br (z' - zs') + kr (z - zs) = F(t)` and
    `Ms zs'' + br (zs' - z') + bs zs' + kr (zs - z) + ks zs + k2 (alpha zs^2 + β zs^3) = 0`. The load is two rectangular
    pulses of force `F` per period `T`, one per wheel of a bogie, `spacing` apart. The nonlinear ballast term, with
    `alpha = 1.75` and `β = -1`, is off for `k2 = 0`.

    Args:
        F:
            The force of a wheel.

        T:
            The period of the passing bogies.

        width:
            The duration of the load of one wheel.

        spacing:
            The time between the two wheels of a bogie.

        k2:
            The stiffness of the nonlinear ballast term.

        Mr:
            The mass of the rail segment.

        Ms:
            The mass of the sleeper.

    Returns:
        system:
            The model, with the deflections `rail.z` of the rail and `rail.zs` of the sleeper, and their velocities.

    Examples:
        >>> from skcomplex.models.engineering.power_systems import wheel_rail_contact
        >>> wheel_rail_contact(k2=500.0).elements[0].k2
        500.0
    """
    rail = RailSegment('rail', F=F, T=T, width=width, spacing=spacing, k2=k2, Mr=Mr, Ms=Ms)
    return System(components=[rail])
