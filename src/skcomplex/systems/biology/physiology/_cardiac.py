"""Physiology systems: electrical waves in cardiac tissue."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import DiffusionInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.engine.solvers import RK4Solver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system


class AlievPanfilovElement(Element):
    """Cardiac tissue of Aliev and Panfilov, an excitable medium with the restitution of heart muscle.

    The potential follows `u' = -k u (u - a) (u - 1) - u v` and the recovery `v' = (epsilon0 + mu1 v / (u + mu2))
    (-v - k u (u - a - 1))`, and a diffusion of the potential couples the cells of the tissue. The recovery rate
    depends on the state, so the action potential lasts longer than in the FitzHugh-Nagumo medium and its duration
    adapts to the pacing rate.

    Args:
        label:
            The label of the tissue.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        u:
            The initial potential.

        v:
            The initial recovery.

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
        >>> from skcomplex.systems.biology.physiology import AlievPanfilovElement
        >>> AlievPanfilovElement('tissue').k
        8.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        v: Annotated[float, StateSpec()] = 0.0,
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 8.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.15,
        epsilon0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.002,
        mu1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        mu2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.v = v
        self.k = k
        self.a = a
        self.epsilon0 = epsilon0
        self.mu1 = mu1
        self.mu2 = mu2

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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
            RateLaw(s.u, -s.k * s.u * (s.u - s.a) * (s.u - 1) - s.u * s.v),
            RateLaw(s.v, recovery_rate * (-s.v - s.k * s.u * (s.u - s.a - 1))),
        ]


@register_system(
    field='biology/physiology',
    tags=('field', 'excitable media', 'spirals', 'cardiac', 'pde'),
    references=('Aliev and Panfilov 1996', 'Datseris and Parlitz 2022'),
    defaults={'until': 200.0, 'dt': 2.0, 'solver': RK4Solver(dt=0.05)},
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
    """Return the Aliev-Panfilov system of cardiac tissue, started from a broken wave that curls into a spiral.

    A spiral wave in heart muscle is a re-entrant arrhythmia, a tachycardia, and its breakup into many waves systems
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
            The system, with the potential `tissue.u` and the recovery `tissue.v`.

    Examples:
        >>> from skcomplex.systems.biology.physiology import aliev_panfilov
        >>> aliev_panfilov((20, 20)).elements[0].values['k']
        8.0
    """
    rows, columns = shape
    u, v = np.zeros(shape), np.zeros(shape)
    u[: rows // 2, :] = 1.0
    v[:, : columns // 2] = 2.0
    values = {'u': u, 'v': v, 'k': k, 'a': a, 'epsilon0': epsilon0, 'mu1': mu1, 'mu2': mu2}
    space = LatticeSpace((rows, columns), boundary='reflecting')
    tissue = FieldElement('tissue', element=AlievPanfilovElement, space=space, values=values)
    diffusion = DiffusionInteraction('diffusion', target='tissue', state='u', coefficient=D)
    return System(elements=[tissue], interactions=[diffusion])
