"""Physiology models: electrical waves in cardiac tissue."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np

from ....core.base import Domain, parameter, state
from ....core.components import Element, Field
from ....core.laws import Law, Rate, Symbols, laplacian
from ....core.runtime import register_model
from ....core.solvers import RK4
from ....core.space import LatticeSpace
from ....core.system import System


class AlievPanfilov(Element):
    """Cardiac tissue of Aliev and Panfilov, an excitable medium with the restitution of heart muscle.

    The potential follows `u' = D Δu - k u (u - a) (u - 1) - u v` and the recovery `v' = (epsilon0 + mu1 v / (u +
    mu2)) (-v - k u (u - a - 1))`. The recovery rate depends on the state, so the action potential lasts longer than
    in the FitzHugh-Nagumo medium and its duration adapts to the pacing rate.

    Args:
        label:
            The label of the tissue.

        u:
            The initial potential.

        v:
            The initial recovery.

        D:
            The diffusion of the potential.

        k:
            The strength of the excitation.

        a:
            The excitation threshold.

        epsilon0:
            The recovery rate far from the excited state.

        mu1:
            The growth of the recovery rate with the recovery.

        mu2:
            The offset of the recovery rate.

    Examples:
        >>> from skcomplex.models.biology.physiology import AlievPanfilov
        >>> AlievPanfilov('tissue').k
        8.0
    """

    u: float = state(default=0.0)
    v: float = state(default=0.0)
    D: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    k: float = parameter(default=8.0, domain=Domain.POSITIVE)
    a: float = parameter(default=0.15, domain=Domain.POSITIVE)
    epsilon0: float = parameter(default=0.002, domain=Domain.POSITIVE)
    mu1: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    mu2: float = parameter(default=0.3, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the diffusing potential and its state-dependent recovery.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        recovery_rate = s.epsilon0 + s.mu1 * s.v / (s.u + s.mu2)
        return [
            Rate(s.u, s.D * laplacian(s.u) - s.k * s.u * (s.u - s.a) * (s.u - 1) - s.u * s.v),
            Rate(s.v, recovery_rate * (-s.v - s.k * s.u * (s.u - s.a - 1))),
        ]


@register_model(
    field='biology/physiology',
    tags=('field', 'excitable media', 'spirals', 'cardiac', 'pde'),
    references=('Aliev and Panfilov 1996', 'Datseris and Parlitz 2022'),
    defaults={'until': 200.0, 'dt': 2.0, 'solver': RK4(dt=0.05)},
)
def aliev_panfilov(
    shape: Sequence[int] = (100, 100),
    *,
    D: float = 1.0,
    k: float = 8.0,
    a: float = 0.15,
    epsilon0: float = 0.002,
    mu1: float = 0.2,
    mu2: float = 0.3,
) -> System:
    """Return the Aliev-Panfilov model of cardiac tissue, started from a broken wave that curls into a spiral.

    A spiral wave in heart muscle is a re-entrant arrhythmia, a tachycardia, and its breakup into many waves models
    fibrillation. The upper half of the tissue starts excited and its left half refractory, so the free end of the
    wave rotates.

    Args:
        shape:
            The number of rows and columns of the lattice with zero-flux edges.

        D:
            The diffusion of the potential.

        k:
            The strength of the excitation.

        a:
            The excitation threshold.

        epsilon0:
            The recovery rate far from the excited state.

        mu1:
            The growth of the recovery rate with the recovery.

        mu2:
            The offset of the recovery rate.

    Returns:
        system:
            The model, with the potential `tissue.u` and the recovery `tissue.v`.

    Examples:
        >>> from skcomplex.models.biology.physiology import aliev_panfilov
        >>> aliev_panfilov((20, 20)).elements[0].values['k']
        8.0
    """
    rows, columns = shape
    u, v = np.zeros(shape), np.zeros(shape)
    u[: rows // 2, :] = 1.0
    v[:, : columns // 2] = 2.0
    values = {'u': u, 'v': v, 'D': D, 'k': k, 'a': a, 'epsilon0': epsilon0, 'mu1': mu1, 'mu2': mu2}
    space = LatticeSpace((rows, columns), boundary='reflecting')
    return System(components=[Field('tissue', AlievPanfilov, space, values=values)])
