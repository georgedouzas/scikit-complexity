"""Electromechanical energy systems: braking recovery, chargers, and loaded rails.

The braking recovers energy, and the chargers are homopolar.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import derivative, pulse_train, sign
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec, VisualSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations
from ..common.mechanics import SpringDamperInteraction


class RailSegmentElement(Element):
    """Rail segment on a sleeper, loaded by passing wheels.

    It is two masses joined by a spring and a damper. The element holds the wheel load and the ballast under the
    sleeper, and the system joins the rail and the sleeper by a `SpringDamperInteraction` of the stiffness `kr` and the
    damping `br`.
    """

    law_classes = (RateLaw,)

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
        """Return the accelerations of the rail and of the sleeper.

        The wheels accelerate the rail and the ballast the sleeper. They leave the pad apart.
        """
        return [
            RateLaw(s.z, RailSegmentElement._get_wheel_load(s) / s.Mr),
            RateLaw(
                s.zs,
                (-s.bs * derivative(s.zs, 1) - s.ks * s.zs - s.k2 * (s.alpha * s.zs**2 + s.beta * s.zs**3)) / s.Ms,
            ),
        ]

    @staticmethod
    def _get_wheel_load(s: SimpleNamespace) -> Expression:
        """Return the force of the two wheels of a bogie.

        It is two pulses per period, `spacing` apart.
        """
        return pulse_train(s.t, s.T, s.width, s.F) + pulse_train(s.t, s.T, s.width, s.F, start=s.spacing)


class RegenerativeBrakingSystem(DynamicalSystem):
    """A car braked by a Faraday disk generator on its wheel.

    The generator charges a supercapacitor.

    The disk generates the voltage `A v`, with the structural constant `A = |B| R^2 / (2 r_w)` of the disk and the
    wheel, and drives the current through an inductor and a resistor into a supercapacitor, `L I' = A v - R I - V_s` and
    `C V_s' = I`. The current brakes the car with the force `A I`, together with dry and viscous friction, `m v' = -A I
    - alpha sign(v) - β v`, so the kinetic energy of the car ends up in the capacitor and in the losses. The system has
    the speed `car.v`, the current `car.I`, and the capacitor voltage `car.Vs`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.power_systems import RegenerativeBrakingSystem
        >>> RegenerativeBrakingSystem(C=100.0).prepare().system_.elements[0].get_value('C')
        100.0
    """

    field = 'engineering/power_systems'
    topics = ('energy', 'electromechanics', 'vehicles', 'ode')
    references = ('Trzaska 2023',)
    until = 60.0
    dt = 0.5

    def __init__(
        self,
        v: float = 20.0,
        *,
        m: float = 1000.0,
        A: float = 1.2,
        L: float = 0.01,
        R: float = 0.05,
        C: float = 680.0,
        alpha: float = 50.0,
        beta: float = 5.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.v = v
        self.m = m
        self.A = A
        self.L = L
        self.R = R
        self.C = C
        self.alpha = alpha
        self.beta = beta

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'car'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'v': DeclaredVariableSpec(self.v, StateSpec()),
            'I': DeclaredVariableSpec(0.0, StateSpec()),
            'Vs': DeclaredVariableSpec(0.0, StateSpec()),
            'm': DeclaredVariableSpec(self.m, ParameterSpec(domain=RealSet.open(0, oo))),
            'A': DeclaredVariableSpec(self.A, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'L': DeclaredVariableSpec(self.L, ParameterSpec(domain=RealSet.open(0, oo))),
            'R': DeclaredVariableSpec(self.R, ParameterSpec(domain=RealSet.open(0, oo))),
            'C': DeclaredVariableSpec(self.C, ParameterSpec(domain=RealSet.open(0, oo))),
            'alpha': DeclaredVariableSpec(self.alpha, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'beta': DeclaredVariableSpec(self.beta, ParameterSpec(domain=RealSet.closed_open(0, oo))),
        }

    def _get_equations(self) -> Equations:
        """Return the rates of the speed, the current, and the charge of the car."""
        return {
            'v': lambda s: (
                -(s.target.A * s.target.I + s.target.alpha * sign(s.target.v) + s.target.beta * s.target.v) / s.target.m
            ),
            'I': lambda s: (s.target.A * s.target.v - s.target.R * s.target.I - s.target.Vs) / s.target.L,
            'Vs': lambda s: s.target.I / s.target.C,
        }


class FaradayDiskSystem(DynamicalSystem):
    """A battery charger driven by a Faraday disk generator.

    The own current of the generator strengthens its field. A motor turns the conducting disk of radii `r1` and `r2`
    with a constant torque `M` in the field `B` of a magnet plus the field `μ N I / l` of a coil that the generated
    current flows through. With `a = B (r2^2 - r1^2) / 2` and `b = μ N (r2^2 - r1^2) / (2 l)`, the current, the
    polarization voltage of the battery, and the angular speed follow `L I' = -R1 I - U + (a + b I) ω - E(I)`,
    `C U' = I - U / R2`, and `J ω' = M - alpha sign(ω) - β r2 ω - (a + b I) I`, where the battery voltage is
    `E(I) = a0 + a1 I + a3 I^3` and the last term is the braking torque of the Lorentz force.

    The generator charges the battery only when its voltage `a ω` exceeds the open-circuit voltage `a0 = 8` of the
    battery, at `ω` above 35 for the default field. Below that speed the steady current is a small negative current,
    the battery discharges through the disk and helps the motor, and the steady speed grows linearly with the torque,
    as `(M - alpha) / (β r2)` up to the small Lorentz torque: with the defaults the current settles at -0.040 and the
    speed at 15.7. The coil acts as a negative resistance `b ω` against `R1 + a1`, so above `ω = (R1 + a1) / b`, about
    32 or a torque near 9.5 for the defaults, the steady state loses its stability and the current oscillates around
    zero, swinging between charging and discharging, while the mean braking torque `b <I²>` bends the speed below the
    linear trend. A stronger magnet charges at the default torque: `B = 4` gives a steady current of 0.013.

    The system has the current `charger.I`, the polarization voltage `charger.U`, and the angular speed `charger.omega`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.power_systems import FaradayDiskSystem
        >>> FaradayDiskSystem(M=6.0).prepare().system_.elements[0].get_value('M')
        6.0
    """

    field = 'engineering/power_systems'
    topics = ('energy', 'electromechanics', 'batteries', 'ode')
    references = ('Trzaska 2023',)
    until = 50.0
    dt = 0.1

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.I = I
        self.U = U
        self.omega = omega
        self.B = B
        self.M = M
        self.N = N
        self.R1 = R1
        self.R2 = R2
        self.L = L
        self.C = C

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'charger'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'I': DeclaredVariableSpec(self.I, StateSpec()),
            'U': DeclaredVariableSpec(self.U, StateSpec()),
            'omega': DeclaredVariableSpec(self.omega, StateSpec()),
            'B': DeclaredVariableSpec(self.B, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'r1': DeclaredVariableSpec(0.025, ParameterSpec(domain=RealSet.open(0, oo))),
            'r2': DeclaredVariableSpec(0.55, ParameterSpec(domain=RealSet.open(0, oo))),
            'L': DeclaredVariableSpec(self.L, ParameterSpec(domain=RealSet.open(0, oo))),
            'C': DeclaredVariableSpec(self.C, ParameterSpec(domain=RealSet.open(0, oo))),
            'R1': DeclaredVariableSpec(self.R1, ParameterSpec(domain=RealSet.open(0, oo))),
            'R2': DeclaredVariableSpec(self.R2, ParameterSpec(domain=RealSet.open(0, oo))),
            'alpha': DeclaredVariableSpec(0.5, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'beta': DeclaredVariableSpec(0.5, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'J': DeclaredVariableSpec(0.5, ParameterSpec(domain=RealSet.open(0, oo))),
            'mu': DeclaredVariableSpec(0.00012566370614359174, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'N': DeclaredVariableSpec(self.N, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'l': DeclaredVariableSpec(0.05, ParameterSpec(domain=RealSet.open(0, oo))),
            'M': DeclaredVariableSpec(self.M, ParameterSpec()),
            'a0': DeclaredVariableSpec(8.0, ParameterSpec()),
            'a1': DeclaredVariableSpec(0.75, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'a3': DeclaredVariableSpec(0.5, ParameterSpec(domain=RealSet.closed_open(0, oo))),
        }

    def _get_equations(self) -> Equations:
        """Return the rates of the spin, the current, and the charge of the disk."""
        return {
            'I': lambda s: (
                (
                    -s.target.R1 * s.target.I
                    - s.target.U
                    + (s.target.B + s.target.mu * s.target.N * s.target.I / s.target.l)
                    * (s.target.r2**2 - s.target.r1**2)
                    * s.target.omega
                    / 2
                    - (s.target.a0 + s.target.a1 * s.target.I + s.target.a3 * s.target.I**3)
                )
                / s.target.L
            ),
            'U': lambda s: (s.target.I - s.target.U / s.target.R2) / s.target.C,
            'omega': lambda s: (
                (
                    s.target.M
                    - s.target.alpha * sign(s.target.omega)
                    - s.target.beta * s.target.r2 * s.target.omega
                    - s.target.I
                    * (s.target.B + s.target.mu * s.target.N * s.target.I / s.target.l)
                    * (s.target.r2**2 - s.target.r1**2)
                    / 2
                )
                / s.target.J
            ),
        }


class WheelRailContactSystem(BaseSystem):
    """A rail segment on a sleeper, loaded by the wheels of passing bogies.

    The rail and the sleeper are lumped masses joined by the pad, a spring and a damper, and the sleeper rests on the
    ballast, `Mr z'' + br (z' - zs') + kr (z - zs) = F(t)` and `Ms zs'' + br (zs' - z') + bs zs' + kr (zs - z) + ks zs +
    k2 (alpha zs^2 + β zs^3) = 0`. The load is two rectangular pulses of force `F` per period `T`, one per wheel of a
    bogie, `spacing` apart. The nonlinear ballast term, with `alpha = 1.75` and `β = -1`, is off for `k2 = 0`. The
    system has the deflections `rail.z` of the rail and `rail.zs` of the sleeper, and their velocities.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.power_systems import WheelRailContactSystem
        >>> WheelRailContactSystem(k2=500.0).prepare().system_.elements[0].k2
        500.0
    """

    field = 'engineering/power_systems'
    topics = ('energy', 'vibration', 'railways', 'pulses', 'ode')
    references = ('Trzaska 2023',)
    until = 60.0
    dt = 0.05

    def __init__(
        self,
        *,
        F: float = 1000.0,
        T: float = 10.0,
        width: float = 1.0,
        spacing: float = 3.0,
        k2: float = 0.0,
        Mr: float = 60.0,
        Ms: float = 156.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.F = F
        self.T = T
        self.width = width
        self.spacing = spacing
        self.k2 = k2
        self.Mr = Mr
        self.Ms = Ms

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        rail = RailSegmentElement(
            'rail',
            F=self.F,
            T=self.T,
            width=self.width,
            spacing=self.spacing,
            k2=self.k2,
            Mr=self.Mr,
            Ms=self.Ms,
        )
        pad = SpringDamperInteraction(
            'pad',
            target='rail',
            states=('z', 'zs'),
            stiffness='kr',
            damping='br',
            divisor={'z': 'Mr', 'zs': 'Ms'},
        )
        return CompositeSystem(self.label, elements=[rail], interactions=[pad])
