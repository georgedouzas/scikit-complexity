"""Low-order systems of the atmosphere and the geodynamo."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, vector

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.spaces import ContinuousSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class Lorenz84Element(Element):
    """Atmosphere of Lorenz-84.

    The general circulation is reduced to a westerly jet `x` and a travelling wave of eddies, whose cosine and sine
    phases are `y` and `z`. `Lorenz84Interaction` gives their laws.

    Args:
        label:
            The label of the atmosphere.

        x:
            The initial strength of the westerly jet.

        y:
            The initial cosine phase of the eddies.

        z:
            The initial sine phase of the eddies.

    Examples:
        >>> from skcomplex.systems.physics.climate import Lorenz84Element
        >>> Lorenz84Element('atmosphere', x=2.0).x
        2.0
    """

    def __init__(self, label: str, *, x: float = 1.0, y: float = 1.0, z: float = 1.0) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.z = z


class Lorenz84Interaction(Interaction):
    """Forcing and damping of the jet of Lorenz-84 and its exchange with the eddies.

    The jet `x` and the phases `y` and `z` of the eddies of the target follow `x' = -y^2 - z^2 - a x + a F`,
    `y' = x y - b x z - y + G`, and `z' = b x y + x z - z`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the atmosphere.

        a:
            The damping of the jet relative to the eddies.

        b:
            The displacement of the eddies by the jet.

        F:
            The forcing of the jet by the equator-to-pole temperature contrast.

        G:
            The forcing of the eddies by the land-sea contrast.

    Examples:
        >>> from skcomplex.systems.physics.climate import Lorenz84Interaction
        >>> Lorenz84Interaction('circulation', target='atmosphere').F
        8.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        a: float = 0.25,
        b: float = 4.0,
        F: float = 8.0,
        G: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.a = a
        self.b = b
        self.F = F
        self.G = G

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rates of the jet and of both phases of the eddies.

        Args:
            context:
                The namespace of the atmosphere, the time `t`, and the parameters.
        """
        super()._prepare(context)
        x, y, z, t = context.target.x, context.target.y, context.target.z, context.t
        a, b = context.a, context.b
        self.laws_ = [
            RateLaw(x.diff(t), -(y**2) - z**2 - a * x + a * context.F),
            RateLaw(y.diff(t), x * y - b * x * z - y + context.G),
            RateLaw(z.diff(t), b * x * y + x * z - z),
        ]


class Lorenz96Element(Element):
    """Ring of Lorenz-96 sites around a latitude circle, with the values `x` of all sites.

    Args:
        label:
            The label of the ring.

        x:
            The initial values of all sites.

    Examples:
        >>> from skcomplex.systems.physics.climate import Lorenz96Element
        >>> Lorenz96Element('ring', x=[8.0, 8.0, 8.0, 8.0]).x
        [8.0, 8.0, 8.0, 8.0]
    """

    def __init__(self, label: str, *, x: npt.ArrayLike = 8.0) -> None:
        super().__init__(label)
        self.x = x


class Lorenz96Interaction(Interaction):
    """Advection, damping, and forcing of the sites of a Lorenz-96 ring.

    Every site follows `dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F`, with the indices taken around the ring.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the ring, whose value `x` holds every site.

        F:
            The constant forcing.

    Examples:
        >>> from skcomplex.systems.physics.climate import Lorenz96Interaction
        >>> Lorenz96Interaction('advection', target='ring').F
        8.0
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role, F: float = 8.0) -> None:
        super().__init__(label)
        self.target = target
        self.F = F

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rate of the values of all sites.

        Args:
            context:
                The namespace of the ring, the time `t`, and the forcing.
        """
        super()._prepare(context)
        x = context.target.x
        n_sites = len(x)
        rates = [(x[(i + 1) % n_sites] - x[i - 2]) * x[i - 1] - x[i] + context.F for i in range(n_sites)]
        self.laws_ = [RateLaw(x, vector(SR, rates))]


class Lorenz84System(BaseSystem):
    """Lorenz-84, the general circulation as a westerly jet and a travelling wave.

    The system is chaotic for `F = 8`.

    The jet `x` and the cosine and sine phases `y` and `z` of the eddies follow `dx/dt = -y^2 - z^2 - a x + a F`, `dy/dt
    = x y - b x z - y + G`, and `dz/dt = b x y + x z - z`. The system has the states `atmosphere.x`, `atmosphere.y`, and
    `atmosphere.z`.

    Args:
        x:
            The initial strength of the westerly jet.

        y:
            The initial cosine phase of the eddies.

        z:
            The initial sine phase of the eddies.

        a:
            The damping of the jet relative to the eddies.

        b:
            The displacement of the eddies by the jet.

        F:
            The forcing of the jet by the equator-to-pole temperature contrast.

        G:
            The forcing of the eddies by the land-sea contrast.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.climate import Lorenz84System
        >>> Lorenz84System(G=1.25).prepare().system_.interactions[0].G
        1.25
    """

    field = 'physics/climate'
    topics = ('chaos', 'ode', 'atmosphere', 'general circulation')
    references = ('Lorenz 1984', 'Datseris and Parlitz 2022')
    until = 100.0
    dt = 0.05

    def __init__(
        self,
        x: float = 1.0,
        y: float = 1.0,
        z: float = 1.0,
        *,
        a: float = 0.25,
        b: float = 4.0,
        F: float = 8.0,
        G: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b
        self.F = F
        self.G = G

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the jet and the eddies of the atmosphere.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Lorenz84Element('atmosphere', x=self.x, y=self.y, z=self.z)],
            interactions=[
                Lorenz84Interaction('circulation', target='atmosphere', a=self.a, b=self.b, F=self.F, G=self.G),
            ],
        )


class Lorenz96System(BaseSystem):
    """Lorenz-96 on a ring of sites around a latitude circle.

    Every site follows `dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F`.

    The sites start at the steady state `F`, and the first one is displaced by `perturbation`. For `F = 8` the steady
    state is unstable and the ring is chaotic. The system has the values `ring.x` of all sites.

    Args:
        n_sites:
            The number of sites, at least 4, the dimension of the state `x`.

        F:
            The constant forcing.

        perturbation:
            The displacement of the first site from the steady state.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.climate import Lorenz96System
        >>> Lorenz96System(6).prepare().system_.space.n_dim
        6
    """

    field = 'physics/climate'
    topics = ('chaos', 'ode', 'atmosphere', 'spatiotemporal chaos', 'data assimilation')
    references = ('Lorenz 1996', 'Datseris and Parlitz 2022')
    until = 20.0
    dt = 0.05

    def __init__(
        self,
        n_sites: int = 20,
        *,
        F: float = 8.0,
        perturbation: float = 0.01,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_sites = n_sites
        self.F = F
        self.perturbation = perturbation

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a ring of sites, at the steady state but for the first one.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        values = np.full(self.n_sites, float(self.F))
        values[0] += self.perturbation
        self.system_ = System(
            self.label,
            elements=[Lorenz96Element('ring', x=values)],
            interactions=[Lorenz96Interaction('advection', target='ring', F=self.F)],
            space=ContinuousSpace(n_dim=self.n_sites),
        )


class GeodynamoInteraction(Interaction):
    """Growth and damping of the fields of Gissinger's geodynamo and the forcing of its flow.

    The quadrupolar field `Q`, the dipolar field `D`, and the velocity `V` of the target follow `Q' = mu Q - V D`,
    `D' = -nu D + V Q`, and `V' = Gamma - V + Q D`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the dynamo.

        mu:
            The growth rate of the quadrupole.

        nu:
            The damping rate of the dipole.

        Gamma:
            The forcing of the flow.

    Examples:
        >>> from skcomplex.systems.physics.climate import GeodynamoInteraction
        >>> GeodynamoInteraction('dynamics', target='dynamo').mu
        0.119
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        mu: float = 0.119,
        nu: float = 0.1,
        Gamma: float = 0.9,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.mu = mu
        self.nu = nu
        self.Gamma = Gamma

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rates of both fields and of the velocity.

        Args:
            context:
                The namespace of the dynamo, the time `t`, and the parameters.
        """
        super()._prepare(context)
        Q, D, V, t = context.target.Q, context.target.D, context.target.V, context.t
        self.laws_ = [
            RateLaw(Q.diff(t), context.mu * Q - V * D),
            RateLaw(D.diff(t), -context.nu * D + V * Q),
            RateLaw(V.diff(t), context.Gamma - V + Q * D),
        ]


class GeodynamoSystem(BaseSystem):
    """Gissinger's geodynamo, whose dipole field reverses at irregular times.

    The quadrupolar field `Q`, the dipolar field `D`, and the velocity `V` follow `dQ/dt = mu Q - V D`, `dD/dt = -nu D +
    V Q`, and `dV/dt = Gamma - V + Q D`. The equations are symmetric under `(Q, D) -> (-Q, -D)`, so the chaotic flow
    visits both polarities of the dipole. The system has the states `dynamo.Q`, `dynamo.D`, and `dynamo.V`.

    Args:
        Q:
            The initial quadrupolar field.

        D:
            The initial dipolar field.

        V:
            The initial velocity.

        mu:
            The growth rate of the quadrupole.

        nu:
            The damping rate of the dipole.

        Gamma:
            The forcing of the flow.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.climate import GeodynamoSystem
        >>> GeodynamoSystem(mu=0.12).prepare().system_.interactions[0].mu
        0.12
    """

    field = 'physics/climate'
    topics = ('chaos', 'ode', 'geomagnetism', 'reversals')
    references = ('Gissinger 2012', 'Datseris and Parlitz 2022')
    until = 500.0
    dt = 0.5

    def __init__(
        self,
        Q: float = 0.1,
        D: float = 0.1,
        V: float = 0.1,
        *,
        mu: float = 0.119,
        nu: float = 0.1,
        Gamma: float = 0.9,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.Q = Q
        self.D = D
        self.V = V
        self.mu = mu
        self.nu = nu
        self.Gamma = Gamma

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the fields and the flow of the dynamo.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('dynamo', values={'Q': self.Q, 'D': self.D, 'V': self.V})],
            interactions=[GeodynamoInteraction('dynamics', target='dynamo', mu=self.mu, nu=self.nu, Gamma=self.Gamma)],
        )
