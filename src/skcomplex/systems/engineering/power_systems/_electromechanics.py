"""Electromechanical energy systems: braking recovery, chargers, and loaded rails.

The braking recovers energy, and the chargers are homopolar.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import pulse_train, sign
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import ParameterSpec, Role
from ....core.systems import BaseSystem, System
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations
from ..common.mechanics import SpringDamperInteraction


class WheelLoadInteraction(Interaction):
    """Load of the two wheels of passing bogies on a rail.

    Every bogie presses the rail with two rectangular pulses of force `F` and duration `width`, `spacing` apart, once
    per period `T`, and the load accelerates the rail divided by its mass.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element that holds the rail.

        state:
            The name of the deflection of the rail.

        mass:
            The name of the mass of the rail.

        F:
            The force of a wheel.

        T:
            The period of the passing bogies.

        width:
            The duration of the load of one wheel.

        spacing:
            The time between the two wheels of a bogie.

    Examples:
        >>> from skcomplex.systems.engineering.power_systems import WheelLoadInteraction
        >>> WheelLoadInteraction('wheels', target='rail', F=500.0).F
        500.0
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str = 'z',
        mass: str = 'Mr',
        F: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        T: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        width: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        spacing: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.mass = mass
        self.F = F
        self.T = T
        self.width = width
        self.spacing = spacing

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of the rail by the wheels."""
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        s, z = context, getattr(context.target, self.state)
        load = pulse_train(s.t, s.T, s.width, s.F) + pulse_train(s.t, s.T, s.width, s.F, start=s.spacing)
        self.laws_: list[BaseComponent] = [RateLaw(z.diff(s.t, 2), load / getattr(s.target, self.mass))]


class BallastInteraction(Interaction):
    """Ballast under a sleeper, a damped spring with an optional nonlinear term.

    The ballast pushes the sleeper with `-bs zs' - ks zs - k2 (alpha zs^2 + beta zs^3)`, which accelerates it divided
    by its mass. The nonlinear term is off for `k2 = 0`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element that holds the sleeper.

        state:
            The name of the deflection of the sleeper.

        mass:
            The name of the mass of the sleeper.

        ks:
            The stiffness of the ballast.

        bs:
            The damping of the ballast.

        k2:
            The stiffness of the nonlinear term.

        alpha:
            The quadratic coefficient of the nonlinear term.

        beta:
            The cubic coefficient of the nonlinear term.

    Examples:
        >>> from skcomplex.systems.engineering.power_systems import BallastInteraction
        >>> BallastInteraction('ballast', target='rail', k2=500.0).k2
        500.0
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str = 'zs',
        mass: str = 'Ms',
        ks: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1500.0,
        bs: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 520.0,
        k2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        alpha: float = 1.75,
        beta: float = -1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.mass = mass
        self.ks = ks
        self.bs = bs
        self.k2 = k2
        self.alpha = alpha
        self.beta = beta

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of the sleeper by the ballast."""
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        s, zs = context, getattr(context.target, self.state)
        force = -s.bs * zs.diff(s.t) - s.ks * zs - s.k2 * (s.alpha * zs**2 + s.beta * zs**3)
        self.laws_: list[BaseComponent] = [RateLaw(zs.diff(s.t, 2), force / getattr(s.target, self.mass))]


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
        >>> RegenerativeBrakingSystem(C=100.0).prepare().system_.elements[0].values['C']
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
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

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state by name."""
        return {
            'v': self.v,
            'I': 0.0,
            'Vs': 0.0,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the value of every parameter by name."""
        return {
            'm': self.m,
            'A': self.A,
            'L': self.L,
            'R': self.R,
            'C': self.C,
            'alpha': self.alpha,
            'beta': self.beta,
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
        >>> FaradayDiskSystem(M=6.0).prepare().system_.elements[0].values['M']
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
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

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state by name."""
        return {
            'I': self.I,
            'U': self.U,
            'omega': self.omega,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the value of every parameter by name."""
        return {
            'B': self.B,
            'r1': 0.025,
            'r2': 0.55,
            'L': self.L,
            'C': self.C,
            'R1': self.R1,
            'R2': self.R2,
            'alpha': 0.5,
            'beta': 0.5,
            'J': 0.5,
            'mu': 0.00012566370614359174,
            'N': self.N,
            'l': 0.05,
            'M': self.M,
            'a0': 8.0,
            'a1': 0.75,
            'a3': 0.5,
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
    system has the deflections `rail.z` of the rail and `rail.zs` of the sleeper, and their
    velocities `rail.z'` and `rail.zs'`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.engineering.power_systems import WheelRailContactSystem
        >>> WheelRailContactSystem(k2=500.0).prepare().system_.interactions[1].k2
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.F = F
        self.T = T
        self.width = width
        self.spacing = spacing
        self.k2 = k2
        self.Mr = Mr
        self.Ms = Ms

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the rail and its sleeper, the wheels, the pad, and the ballast."""
        super()._prepare(context)
        rail = Element('rail', values={'z': 0.0, 'zs': 0.0, 'Mr': self.Mr, 'Ms': self.Ms})
        wheels = WheelLoadInteraction(
            'wheels',
            target='rail',
            F=self.F,
            T=self.T,
            width=self.width,
            spacing=self.spacing,
        )
        ballast = BallastInteraction('ballast', target='rail', k2=self.k2)
        pad = SpringDamperInteraction('pad', target='rail', states=('z', 'zs'), masses=('Mr', 'Ms'), k=6000.0, b=300.0)
        self.system_ = System(self.label, elements=[rail], interactions=[wheels, ballast, pad])
