"""Energy harvesters: circuits that turn vibrations into electric energy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import Expression, oo, sin
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import ParameterSpec, Role
from ....core.systems import BaseSystem, System
from ..common.mechanics import SpringDamperInteraction


class RoadInteraction(Interaction):
    """Tyre between the unsprung mass of a car and a road of the profile `z(t) = A sin(omega t)`.

    The tyre pushes the mass with the force `-k1 (y1 - z(t))`, which accelerates it divided by the mass.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element that holds the unsprung mass.

        state:
            The name of the displacement of the unsprung mass.

        mass:
            The name of the unsprung mass.

        k1:
            The stiffness of the tyre.

        A:
            The amplitude of the road profile.

        omega:
            The angular frequency of the road profile.

    Examples:
        >>> from skcomplex.systems.engineering.circuits._harvesters import RoadInteraction
        >>> RoadInteraction('road', target='harvester', A=0.02).A
        0.02
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str = 'y1',
        mass: str = 'm1',
        k1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 200000.0,
        A: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        omega: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 9.42,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.mass = mass
        self.k1 = k1
        self.A = A
        self.omega = omega

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of the unsprung mass by the tyre on the road."""
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        y1, t = getattr(context.target, self.state), context.t
        force = -context.k1 * (y1 - context.A * sin(context.omega * t))
        self.laws_: list[BaseComponent] = [RateLaw(y1.diff(t, 2), force / getattr(context.target, self.mass))]


def _get_piezo_voltage(s: SimpleNamespace) -> Expression:
    """Return the voltage of the piezoelectric bar, `n d k2 (y2 - y1) / C`."""
    harvester = s.harvester
    return harvester.n * harvester.d * s.suspension.k * (harvester.y2 - harvester.y1) / harvester.C


class VibrationEnergyHarvesterSystem(BaseSystem):
    """A dual-mass piezoelectric harvester in a car suspension.

    The roughness of the road drives it. The unsprung and sprung masses follow
    `m1 y1'' + b (y1' - y2') + k2 (y1 - y2) + k1 (y1 - z(t)) = 0` and `m2 y2'' + b (y2' - y1') + k2 (y2 - y1) = 0`, with
    the road profile `z(t) = A sin(ω t)`. A lever with arm ratio `n` presses a piezoelectric bar of constant `d` and
    capacitance `C`, whose charge is `n d k2 (y2 - y1)` and whose voltage is the observable `voltage`. The damper `b`
    stands for the energy the bar absorbs. The system has the displacements `harvester.y1` and `harvester.y2`, and the
    observable `voltage`.

    Args:
        m1:
            The unsprung mass.

        m2:
            The sprung mass.

        k1:
            The stiffness of the tyre.

        k2:
            The stiffness of the suspension spring.

        b:
            The damping coefficient of the harvester.

        A:
            The amplitude of the road profile.

        omega:
            The angular frequency of the road profile.

        n:
            The ratio of the arms of the lever.

        d:
            The piezoelectric constant of the bar.

        C:
            The capacitance of the bar.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import VibrationEnergyHarvesterSystem
        >>> VibrationEnergyHarvesterSystem(b=1000.0).prepare().system_.interactions[1].b
        1000.0
    """

    field = 'engineering/circuits'
    topics = ('energy harvesting', 'piezoelectric', 'vibration', 'vehicles', 'ode')
    references = ('Trzaska 2023',)
    until = 10.0
    dt = 0.01
    observables = (Observable('voltage', _get_piezo_voltage),)

    def __init__(
        self,
        *,
        m1: float = 40.0,
        m2: float = 400.0,
        k1: float = 2.0e5,
        k2: float = 2.0e4,
        b: float = 1500.0,
        A: float = 0.01,
        omega: float = 9.42,
        n: float = 10.0,
        d: float = 4.0e-10,
        C: float = 1.0e-8,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.A = A
        self.omega = omega
        self.n = n
        self.d = d
        self.C = C

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        values = {'y1': 0.0, 'y2': 0.0, 'm1': self.m1, 'm2': self.m2, 'n': self.n, 'd': self.d, 'C': self.C}
        suspension = SpringDamperInteraction(
            'suspension',
            target='harvester',
            states=('y1', 'y2'),
            masses=('m1', 'm2'),
            k=self.k2,
            b=self.b,
        )
        road = RoadInteraction('road', target='harvester', k1=self.k1, A=self.A, omega=self.omega)
        self.system_ = System(
            self.label,
            elements=[Element('harvester', values=values)],
            interactions=[road, suspension],
        )
