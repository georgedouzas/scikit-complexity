"""Low-order models of the atmosphere and the geodynamo."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.base import state, vector
from ....core.model import Element, EuclideanSpace, Law, Rate, Symbols, System
from ....core.runtime import register_model


def _lorenz_96_laws(s: Symbols) -> list[Law]:
    """Return the advection, damping, and forcing of every site of the Lorenz-96 ring."""
    x = s.x
    n_sites = len(x)
    rates = [(x[(i + 1) % n_sites] - x[i - 2]) * x[i - 1] - x[i] + s.F for i in range(n_sites)]
    return [Rate(s.x, vector(rates))]


Lorenz84 = Element.define(
    'Lorenz84',
    states={'x': 1.0, 'y': 1.0, 'z': 1.0},
    parameters={'a': 0.25, 'b': 4.0, 'F': 8.0, 'G': 1.0},
    laws=lambda s: [
        Rate(s.x, -(s.y**2) - s.z**2 - s.a * s.x + s.a * s.F),
        Rate(s.y, s.x * s.y - s.b * s.x * s.z - s.y + s.G),
        Rate(s.z, s.b * s.x * s.y + s.x * s.z - s.z),
    ],
)

Lorenz96 = Element.define(
    'Lorenz96',
    states={'x': state(default=8.0, vector=True)},
    parameters={'F': 8.0},
    laws=_lorenz_96_laws,
)

Geodynamo = Element.define(
    'Geodynamo',
    states={'Q': 0.1, 'D': 0.1, 'V': 0.1},
    parameters={'mu': 0.119, 'nu': 0.1, 'Gamma': 0.9},
    laws=lambda s: [
        Rate(s.Q, s.mu * s.Q - s.V * s.D),
        Rate(s.D, -s.nu * s.D + s.V * s.Q),
        Rate(s.V, s.Gamma - s.V + s.Q * s.D),
    ],
)


@register_model(
    field='earth/climate',
    tags=('chaos', 'ode', 'atmosphere', 'general circulation'),
    references=('Lorenz 1984', 'Datseris and Parlitz 2022'),
    defaults={'until': 100.0, 'dt': 0.05},
)
def lorenz_84(
    x: float = 1.0,
    y: float = 1.0,
    z: float = 1.0,
    *,
    a: float = 0.25,
    b: float = 4.0,
    F: float = 8.0,
    G: float = 1.0,
) -> System:
    """Return Lorenz-84, the general circulation reduced to a westerly jet and a travelling wave, chaotic for `F = 8`.

    The jet `x` and the cosine and sine phases `y` and `z` of the eddies follow `dx/dt = -y^2 - z^2 - a x + a F`,
    `dy/dt = x y - b x z - y + G`, and `dz/dt = b x y + x z - z`.

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

    Returns:
        system:
            The model, with the states `atmosphere.x`, `atmosphere.y`, and `atmosphere.z`.

    Examples:
        >>> from skcomplex.models.earth.climate import lorenz_84
        >>> lorenz_84(G=1.25).elements[0].G
        1.25
    """
    return System(components=[Lorenz84('atmosphere', x=x, y=y, z=z, a=a, b=b, F=F, G=G)])


@register_model(
    field='earth/climate',
    tags=('chaos', 'ode', 'atmosphere', 'spatiotemporal chaos', 'data assimilation'),
    references=('Lorenz 1996', 'Datseris and Parlitz 2022'),
    defaults={'until': 20.0, 'dt': 0.05},
)
def lorenz_96(
    n_sites: int = 20,
    *,
    F: float = 8.0,
    perturbation: float = 0.01,
) -> System:
    """Return Lorenz-96, `dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F` on a ring of sites around a latitude circle.

    The sites start at the steady state `F`, and the first one is displaced by `perturbation`. For `F = 8` the
    steady state is unstable and the ring is chaotic.

    Args:
        n_sites:
            The number of sites, at least 4, the dimension of the state `x`.

        F:
            The constant forcing.

        perturbation:
            The displacement of the first site from the steady state.

    Returns:
        system:
            The model, with the values `ring.x` of all sites.

    Examples:
        >>> from skcomplex.models.earth.climate import lorenz_96
        >>> lorenz_96(6).space.n_dim
        6
    """
    values = np.full(n_sites, float(F))
    values[0] += perturbation
    return System(components=[Lorenz96('ring', x=values, F=F)], space=EuclideanSpace(n_sites))


@register_model(
    field='earth/climate',
    tags=('chaos', 'ode', 'geomagnetism', 'reversals'),
    references=('Gissinger 2012', 'Datseris and Parlitz 2022'),
    defaults={'until': 500.0, 'dt': 0.5},
)
def geodynamo(
    Q: float = 0.1,
    D: float = 0.1,
    V: float = 0.1,
    *,
    mu: float = 0.119,
    nu: float = 0.1,
    Gamma: float = 0.9,
) -> System:
    """Return Gissinger's geodynamo, whose dipole field reverses at irregular times.

    The quadrupolar field `Q`, the dipolar field `D`, and the velocity `V` follow `dQ/dt = mu Q - V D`,
    `dD/dt = -nu D + V Q`, and `dV/dt = Gamma - V + Q D`. The equations are symmetric under `(Q, D) -> (-Q, -D)`, so
    the chaotic flow visits both polarities of the dipole.

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

    Returns:
        system:
            The model, with the states `dynamo.Q`, `dynamo.D`, and `dynamo.V`.

    Examples:
        >>> from skcomplex.models.earth.climate import geodynamo
        >>> geodynamo(mu=0.12).elements[0].mu
        0.12
    """
    return System(components=[Geodynamo('dynamo', Q=Q, D=D, V=V, mu=mu, nu=nu, Gamma=Gamma)])
