"""Physiology systems: electrical waves in cardiac tissue and excitable media."""

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
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import RK4Solver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class BarkleyElement(Element):
    """Excitable medium of Barkley, a fast excitation `u` and a slow recovery `w`.

    The excitation has a polynomial nullcline.

    The excitation follows `u' = u (1 - u) (u - (w + b) / a)` and the recovery `w' = epsilon (u^gamma - w)`. A diffusion
    of the excitation couples the sites of the medium. The threshold `(w + b) / a` rises with the recovery, so an
    excited site recovers before it can fire again.

    Args:
        label:
            The label of the medium.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        u:
            The initial excitation.

        w:
            The initial recovery.

        a:
            The inverse slope of the threshold.

        b:
            The offset of the threshold.

        epsilon:
            The ratio of the time scales of the recovery and the excitation.

        gamma:
            The exponent of the excitation in the recovery, 1 for Barkley's model and 3 for the variant of Bär and
            Eiswirth with spiral breakup.

    Examples:
        >>> from skcomplex.systems.biology.physiology import BarkleyElement
        >>> BarkleyElement('medium').a
        0.6
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        w: Annotated[float, StateSpec()] = 0.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.6,
        b: Annotated[float, ParameterSpec()] = 0.05,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.02,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.w = w
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.gamma = gamma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the cubic excitation and its slow recovery.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            RateLaw(s.u, s.u * (1 - s.u) * (s.u - (s.w + s.b) / s.a)),
            RateLaw(s.w, s.epsilon * (s.u**s.gamma - s.w)),
        ]


class AlievPanfilovElement(Element):
    """Cardiac tissue of Aliev and Panfilov, an excitable medium.

    It has the restitution of heart muscle.

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

    law_classes = (RateLaw,)

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


class AlievPanfilovSystem(BaseSystem):
    """The Aliev-Panfilov system of cardiac tissue, started from a broken wave.

    The broken wave curls into a spiral.

    A spiral wave in heart muscle is a re-entrant arrhythmia, a tachycardia, and its breakup into many waves systems
    fibrillation. The upper half of the tissue starts excited and its left half refractory, so the free end of the wave
    rotates. The system has the potential `tissue.u` and the recovery `tissue.v`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.physiology import AlievPanfilovSystem
        >>> AlievPanfilovSystem((20, 20)).prepare().system_.elements[0].values['k']
        8.0
    """

    field = 'biology/physiology'
    topics = ('field', 'excitable media', 'spirals', 'cardiac', 'pde')
    references = ('Aliev and Panfilov 1996', 'Datseris and Parlitz 2022')
    until = 200.0
    dt = 2.0
    solver = RK4Solver(dt=0.05)

    def __init__(
        self,
        shape: Sequence[int] = (100, 100),
        *,
        D: float = 1.0,
        k: float = 8.0,
        a: float = 0.15,
        epsilon0: float = 0.002,
        mu1: float = 0.2,
        mu2: float = 0.3,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.D = D
        self.k = k
        self.a = a
        self.epsilon0 = epsilon0
        self.mu1 = mu1
        self.mu2 = mu2

    def _build_system(self) -> System:
        """Return the parts of the Aliev-Panfilov system of cardiac tissue."""
        rows, columns = self.shape
        u, v = np.zeros(self.shape), np.zeros(self.shape)
        u[: rows // 2, :] = 1.0
        v[:, : columns // 2] = 2.0
        values = {'u': u, 'v': v, 'k': self.k, 'a': self.a, 'epsilon0': self.epsilon0, 'mu1': self.mu1, 'mu2': self.mu2}
        space = LatticeSpace((rows, columns), boundary='reflecting')
        tissue = FieldElement('tissue', element=AlievPanfilovElement, space=space, values=values)
        diffusion = DiffusionInteraction('diffusion', target='tissue', state='u', coefficient=self.D)
        return System(self.label, elements=[tissue], interactions=[diffusion])


class BarkleySystem(BaseSystem):
    """Barkley's excitable medium, started from orthogonal stripes.

    The crossing of the stripes curls into a spiral wave.

    The excitation `u` and the recovery `w` start as orthogonal stripes of zeros and ones, so the excited stripe runs
    into recovering tissue on one side only and its free end rotates. With `(a, b, d, epsilon) = (0.6, 0.05, 1, 0.02)`
    the spiral is stable for `gamma = 1`. With `(0.7, 0.05, 1, 0.08)` and `gamma = 3` it breaks up into spatiotemporal
    chaos. The source runs a domain of 300 by 300, and the default lattice covers 192 by 192 with a spacing of 3, wide
    enough for the stable spiral. The system has the excitation `medium.u` and the recovery `medium.w`.

    Args:
        shape:
            The number of rows and columns of the lattice with zero-flux edges.

        spacing:
            The distance between neighbouring sites.

        d:
            The diffusion of the excitation.

        a:
            The inverse slope of the threshold.

        b:
            The offset of the threshold.

        epsilon:
            The ratio of the time scales of the recovery and the excitation.

        gamma:
            The exponent of the excitation in the recovery.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.physiology import BarkleySystem
        >>> BarkleySystem((20, 20), gamma=3.0).prepare().system_.elements[0].values['gamma']
        3.0
    """

    field = 'biology/physiology'
    topics = ('field', 'excitable media', 'spirals', 'spatiotemporal chaos', 'pde')
    references = ('Barkley 1991', 'Datseris and Parlitz 2022')
    until = 400.0
    dt = 4.0
    solver = RK4Solver(dt=0.2)

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        spacing: float = 3.0,
        d: float = 1.0,
        a: float = 0.6,
        b: float = 0.05,
        epsilon: float = 0.02,
        gamma: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.spacing = spacing
        self.d = d
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.gamma = gamma

    def _build_system(self) -> System:
        """Return the parts of Barkley's excitable medium."""
        rows, columns = self.shape
        u, w = np.zeros(self.shape), np.zeros(self.shape)
        u[:, : columns // 2] = 1.0
        w[: rows // 2, :] = 1.0
        values = {'u': u, 'w': w, 'a': self.a, 'b': self.b, 'epsilon': self.epsilon, 'gamma': self.gamma}
        space = LatticeSpace((rows, columns), boundary='reflecting', spacing=self.spacing)
        medium = FieldElement('medium', element=BarkleyElement, space=space, values=values)
        diffusion = DiffusionInteraction('diffusion', target='medium', state='u', coefficient=self.d)
        return System(self.label, elements=[medium], interactions=[diffusion])
