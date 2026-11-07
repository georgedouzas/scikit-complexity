"""Physiology systems: delayed feedback and excitable tissue."""

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
from ....core.components.laws.utils import delayed, hill
from ....core.components.spaces import LatticeSpace
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import RK4Solver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..common.excitable import FitzHughNagumoInteraction


class FitzHughNagumoElement(Element):
    """Cell of an excitable medium, with a potential `u` and a recovery `v`.

    The cell has no laws of its own. FitzHugh-Nagumo kinetics and a diffusion of the potential change it.

    Args:
        label:
            The label of the cell.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        u:
            The initial potential.

        v:
            The initial recovery.

    Examples:
        >>> from skcomplex.systems.biology.physiology import FitzHughNagumoElement
        >>> FitzHughNagumoElement('cell', u=1.5).u
        1.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        v: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.v = v


class MackeyGlassElement(Element):
    """Concentration `x` of circulating blood cells, made at a delayed Hill rate and lost at a constant rate.

    The concentration follows `x' = beta x(t - tau) / (1 + x(t - tau)^n) - gamma x`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial concentration, also its history.

        beta:
            The production rate.

        gamma:
            The loss rate.

        n:
            The Hill coefficient of the production.

        tau:
            The delay of the production.

    Examples:
        >>> from skcomplex.systems.biology.physiology import MackeyGlassElement
        >>> MackeyGlassElement('blood', tau=6.0).tau
        6.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        beta: Annotated[float, ParameterSpec()] = 0.2,
        gamma: Annotated[float, ParameterSpec()] = 0.1,
        n: Annotated[float, ParameterSpec()] = 10.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 17.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.tau = tau

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the delayed production and the loss of the cells.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        history = delayed(s.x, s.tau)
        return [RateLaw(s.x, s.beta * history * (1 - hill(history, 1, s.n)) - s.gamma * s.x)]


class FitzhughNagumoSystem(BaseSystem):
    """The FitzHugh-Nagumo excitable medium, started from a broken wave that curls into a spiral.

    The upper half starts excited and the left half refractory, so the free end of the wave rotates. The system has the
    potential `medium.u` and the recovery `medium.v`.

    Args:
        shape:
            The number of rows and columns of the lattice with zero-flux edges.

        D:
            The diffusion of the potential `u`.

        a:
            The offset of the recovery.

        b:
            The damping of the recovery.

        epsilon:
            The time scale of the recovery.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.physiology import FitzhughNagumoSystem
        >>> FitzhughNagumoSystem((20, 20)).prepare().system_.interactions[0].epsilon
        0.02
    """

    field = 'biology/physiology'
    topics = ('field', 'excitable media', 'spirals', 'pde')
    references = ('FitzHugh 1961', 'Nagumo, Arimoto, and Yoshizawa 1962', 'Gros 2015')
    until = 200.0
    dt = 2.0
    solver = RK4Solver(dt=0.05)

    def __init__(
        self,
        shape: Sequence[int] = (100, 100),
        *,
        D: float = 1.0,
        a: float = 0.7,
        b: float = 0.8,
        epsilon: float = 0.02,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.D = D
        self.a = a
        self.b = b
        self.epsilon = epsilon

    def _build_system(self) -> System:
        """Return the parts of the system."""
        rows, columns = self.shape
        u, v = np.full(self.shape, -1.2), np.full(self.shape, -0.62)
        u[: rows // 2, :] = 1.5
        v[:, : columns // 2] = 1.0
        space = LatticeSpace((rows, columns), boundary='reflecting')
        medium = FieldElement('medium', element=FitzHughNagumoElement, space=space, values={'u': u, 'v': v})
        kinetics = FitzHughNagumoInteraction(
            'kinetics',
            target='medium',
            fast='u',
            slow='v',
            a=self.a,
            b=self.b,
            epsilon=self.epsilon,
        )
        diffusion = DiffusionInteraction('diffusion', target='medium', state='u', coefficient=self.D)
        return System(self.label, elements=[medium], interactions=[kinetics, diffusion])


class MackeyGlassSystem(BaseSystem):
    """The Mackey-Glass equation `x' = beta x(t - tau) / (1 + x(t - tau)^n) - gamma x` of blood cell production.

    It is periodic for `tau = 6` and chaotic for `tau = 17`. The history before the start is the initial value. The
    system has the concentration `cells.x`.

    Args:
        x:
            The initial concentration, also its history.

        beta:
            The production rate.

        gamma:
            The decay rate.

        n:
            The steepness of the production feedback.

        tau:
            The delay of production.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.physiology import MackeyGlassSystem
        >>> MackeyGlassSystem(tau=6.0).prepare().system_.elements[0].tau
        6.0
    """

    field = 'biology/physiology'
    topics = ('chaos', 'delay', 'physiology')
    references = ('Mackey and Glass 1977', 'Erneux 2009')
    until = 500.0
    dt = 0.5

    def __init__(
        self,
        x: float = 0.5,
        *,
        beta: float = 0.2,
        gamma: float = 0.1,
        n: float = 10.0,
        tau: float = 17.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.tau = tau

    def _build_system(self) -> System:
        """Return the parts of the system."""
        return System(
            self.label,
            elements=[MackeyGlassElement('cells', x=self.x, beta=self.beta, gamma=self.gamma, n=self.n, tau=self.tau)],
        )
