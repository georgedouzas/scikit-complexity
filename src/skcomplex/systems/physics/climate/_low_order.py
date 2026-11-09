"""Low-order systems of the atmosphere and the geodynamo."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, vector

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.spaces import ContinuousSpace
from ....core.components.systems import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class GeodynamoElement(Element):
    """Dynamo of Gissinger's geodynamo.

    The dynamo holds its quadrupolar and dipolar fields and the velocity of the flow.

    The quadrupolar field `Q`, the dipolar field `D`, and the velocity `V` follow `dQ/dt = mu Q - V D`,
    `dD/dt = -nu D + V Q`, and `dV/dt = Gamma - V + Q D`.

    Args:
        label:
            The label of the dynamo.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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

    Examples:
        >>> from skcomplex.systems.physics.climate import GeodynamoElement
        >>> GeodynamoElement('dynamo').Gamma
        0.9
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        Q: Annotated[float, StateSpec()] = 0.1,
        D: Annotated[float, StateSpec()] = 0.1,
        V: Annotated[float, StateSpec()] = 0.1,
        mu: Annotated[float, ParameterSpec()] = 0.119,
        nu: Annotated[float, ParameterSpec()] = 0.1,
        Gamma: Annotated[float, ParameterSpec()] = 0.9,
    ) -> None:
        super().__init__(label, visual=visual)
        self.Q = Q
        self.D = D
        self.V = V
        self.mu = mu
        self.nu = nu
        self.Gamma = Gamma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the growth and damping of both fields and the forcing of the flow.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the quadrupolar field, the dipolar field, and the velocity.
        """
        return [
            RateLaw(s.Q, s.mu * s.Q - s.V * s.D),
            RateLaw(s.D, -s.nu * s.D + s.V * s.Q),
            RateLaw(s.V, s.Gamma - s.V + s.Q * s.D),
        ]


class Lorenz84Element(Element):
    """Atmosphere of Lorenz-84.

    The general circulation is reduced to a westerly jet and a travelling wave of eddies.

    The jet `x` and the cosine and sine phases `y` and `z` of the eddies follow `dx/dt = -y^2 - z^2 - a x + a F`,
    `dy/dt = x y - b x z - y + G`, and `dz/dt = b x y + x z - z`.

    Args:
        label:
            The label of the atmosphere.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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

    Examples:
        >>> from skcomplex.systems.physics.climate import Lorenz84Element
        >>> Lorenz84Element('atmosphere').F
        8.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        y: Annotated[float, StateSpec()] = 1.0,
        z: Annotated[float, StateSpec()] = 1.0,
        a: Annotated[float, ParameterSpec()] = 0.25,
        b: Annotated[float, ParameterSpec()] = 4.0,
        F: Annotated[float, ParameterSpec()] = 8.0,
        G: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b
        self.F = F
        self.G = G

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the forcing and damping of the jet and its exchange with the eddies.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the jet and of both phases of the eddies.
        """
        return [
            RateLaw(s.x, -(s.y**2) - s.z**2 - s.a * s.x + s.a * s.F),
            RateLaw(s.y, s.x * s.y - s.b * s.x * s.z - s.y + s.G),
            RateLaw(s.z, s.b * s.x * s.y + s.x * s.z - s.z),
        ]


class Lorenz96Element(Element):
    """Ring of Lorenz-96 sites around a latitude circle.

    The sites are advected by their neighbours, damped, and forced.

    Every site follows `dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F`, with the indices taken around the ring.

    Args:
        label:
            The label of the ring.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial values of all sites.

        F:
            The constant forcing.

    Examples:
        >>> from skcomplex.systems.physics.climate import Lorenz96Element
        >>> Lorenz96Element('ring').F
        8.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 8.0,
        F: Annotated[float, ParameterSpec()] = 8.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.F = F

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the advection, damping, and forcing of every site of the ring.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate law of the values of all sites.
        """
        x = s.x
        n_sites = len(x)
        rates = [(x[(i + 1) % n_sites] - x[i - 2]) * x[i - 1] - x[i] + s.F for i in range(n_sites)]
        return [RateLaw(s.x, vector(SR, rates))]


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

    Examples:
        >>> from skcomplex.systems.physics.climate import Lorenz84System
        >>> Lorenz84System(G=1.25).prepare().system_.elements[0].G
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
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b
        self.F = F
        self.G = G

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        return System(
            self.label,
            elements=[
                Lorenz84Element('atmosphere', x=self.x, y=self.y, z=self.z, a=self.a, b=self.b, F=self.F, G=self.G),
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
    ) -> None:
        super().__init__(label)
        self.n_sites = n_sites
        self.F = F
        self.perturbation = perturbation

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        values = np.full(self.n_sites, float(self.F))
        values[0] += self.perturbation
        return System(
            self.label,
            elements=[Lorenz96Element('ring', x=values, F=self.F)],
            space=ContinuousSpace(n_dim=self.n_sites),
        )


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

    Examples:
        >>> from skcomplex.systems.physics.climate import GeodynamoSystem
        >>> GeodynamoSystem(mu=0.12).prepare().system_.elements[0].mu
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
    ) -> None:
        super().__init__(label)
        self.Q = Q
        self.D = D
        self.V = V
        self.mu = mu
        self.nu = nu
        self.Gamma = Gamma

    def _build_system(self) -> System:
        """Return Gissinger's geodynamo, whose dipole field reverses at irregular times."""
        return System(
            self.label,
            elements=[
                GeodynamoElement('dynamo', Q=self.Q, D=self.D, V=self.V, mu=self.mu, nu=self.nu, Gamma=self.Gamma),
            ],
        )
