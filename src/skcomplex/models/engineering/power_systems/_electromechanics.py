"""Electromechanical energy systems: braking energy recovery, homopolar chargers, and loaded rails."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import derivative, pulse_train, sign
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ..common.mechanics import SpringDamperInteraction


def _wheel_load(s: SimpleNamespace) -> Expression:
    """Return the force of the two wheels of a bogie: two pulses per period, `spacing` apart."""
    return pulse_train(s.t, s.T, s.width, s.F) + pulse_train(s.t, s.T, s.width, s.F, start=s.spacing)


class RegenerativeCarElement(Element):
    """Car whose motor recovers braking energy, slowed by dry and viscous friction."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[object, StateSpec()] = 20.0,
        I: Annotated[object, StateSpec()] = 0.0,
        Vs: Annotated[object, StateSpec()] = 0.0,
        m: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1000.0,
        A: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.2,
        L: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
        R: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.05,
        C: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 680.0,
        alpha: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 50.0,
        beta: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v
        self.I = I
        self.Vs = Vs
        self.m = m
        self.A = A
        self.L = L
        self.R = R
        self.C = C
        self.alpha = alpha
        self.beta = beta

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the speed, the current, and the charge of the car."""
        return [
            RateLaw(s.v, -(s.A * s.I + s.alpha * sign(s.v) + s.beta * s.v) / s.m),
            RateLaw(s.I, (s.A * s.v - s.R * s.I - s.Vs) / s.L),
            RateLaw(s.Vs, s.I / s.C),
        ]


class FaradayDiskElement(Element):
    """Homopolar generator, a spinning conducting disk in a field that charges a load."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        I: Annotated[object, StateSpec()] = 0.0,
        U: Annotated[object, StateSpec()] = 0.0,
        omega: Annotated[object, StateSpec()] = 0.0,
        B: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
        r1: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.025,
        r2: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.55,
        L: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.2,
        C: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.75,
        R1: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 2.3,
        R2: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 109.0,
        alpha: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        beta: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        J: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        mu: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00012566370614359174,
        N: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 250.0,
        l: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.05,
        M: Annotated[object, ParameterSpec()] = 4.8,
        a0: Annotated[object, ParameterSpec()] = 8.0,
        a1: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.75,
        a3: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.I = I
        self.U = U
        self.omega = omega
        self.B = B
        self.r1 = r1
        self.r2 = r2
        self.L = L
        self.C = C
        self.R1 = R1
        self.R2 = R2
        self.alpha = alpha
        self.beta = beta
        self.J = J
        self.mu = mu
        self.N = N
        self.l = l
        self.M = M
        self.a0 = a0
        self.a1 = a1
        self.a3 = a3

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the spin, the current, and the charge of the disk."""
        return [
            RateLaw(
                s.I,
                (
                    -s.R1 * s.I
                    - s.U
                    + (s.B + s.mu * s.N * s.I / s.l) * (s.r2**2 - s.r1**2) * s.omega / 2
                    - (s.a0 + s.a1 * s.I + s.a3 * s.I**3)
                )
                / s.L,
            ),
            RateLaw(s.U, (s.I - s.U / s.R2) / s.C),
            RateLaw(
                s.omega,
                (
                    s.M
                    - s.alpha * sign(s.omega)
                    - s.beta * s.r2 * s.omega
                    - s.I * (s.B + s.mu * s.N * s.I / s.l) * (s.r2**2 - s.r1**2) / 2
                )
                / s.J,
            ),
        ]


class RailSegmentElement(Element):
    """Rail segment on a sleeper, two masses joined by a spring and a damper and loaded by passing wheels.

    The element holds the wheel load and the ballast under the sleeper, and the model joins the rail and the sleeper
    by a `SpringDamperInteraction` of the stiffness `kr` and the damping `br`.
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        z: Annotated[object, StateSpec(order=2)] = 0.0,
        z_d1: float = 0.0,
        zs: Annotated[object, StateSpec(order=2)] = 0.0,
        zs_d1: float = 0.0,
        Mr: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 60.0,
        Ms: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 156.0,
        kr: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 6000.0,
        br: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 300.0,
        ks: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1500.0,
        bs: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 520.0,
        k2: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        alpha: Annotated[object, ParameterSpec()] = 1.75,
        beta: Annotated[object, ParameterSpec()] = -1.0,
        F: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        T: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        width: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        spacing: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.z = z
        self.z_d1 = z_d1
        self.zs = zs
        self.zs_d1 = zs_d1
        self.Mr = Mr
        self.Ms = Ms
        self.kr = kr
        self.br = br
        self.ks = ks
        self.bs = bs
        self.k2 = k2
        self.alpha = alpha
        self.beta = beta
        self.F = F
        self.T = T
        self.width = width
        self.spacing = spacing

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the accelerations of the rail by the wheels and of the sleeper by the ballast, the pad apart."""
        return [
            RateLaw(s.z, _wheel_load(s) / s.Mr),
            RateLaw(
                s.zs,
                (-s.bs * derivative(s.zs, 1) - s.ks * s.zs - s.k2 * (s.alpha * s.zs**2 + s.beta * s.zs**3)) / s.Ms,
            ),
        ]


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
    car = RegenerativeCarElement('car', v=v, I=0.0, Vs=0.0, m=m, A=A, L=L, R=R, C=C, alpha=alpha, beta=beta)
    return System(elements=[car])


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
    charger = FaradayDiskElement('charger', I=I, U=U, omega=omega, B=B, M=M, N=N, R1=R1, R2=R2, L=L, C=C)
    return System(elements=[charger])


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
    rail = RailSegmentElement('rail', F=F, T=T, width=width, spacing=spacing, k2=k2, Mr=Mr, Ms=Ms)
    pad = SpringDamperInteraction(
        'pad',
        target='rail',
        states=('z', 'zs'),
        stiffness='kr',
        damping='br',
        divisor={'z': 'Mr', 'zs': 'Ms'},
    )
    return System(elements=[rail], interactions=[pad])
