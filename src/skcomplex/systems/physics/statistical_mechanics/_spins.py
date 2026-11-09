"""Statistical mechanics systems.

The module covers gases, fluids, and spin systems.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, Expression, cos, oo, pi
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws.utils import neighbors, where
from ....core.components.spaces import BaseNeighborhood, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..common.spins import IsingSamplingInteraction, IsingSpinElement, SpinSamplingInteraction


class PottsSpinElement(Element):
    """Spin of the q-state Potts system, in a state from 0 to `q - 1`.

    The spin has its coupling `J` and its temperature.

    A `PottsSamplingInteraction` redraws the states by Monte Carlo sweeps at the temperature of the spins.

    Args:
        label:
            The label of the spin or the lattice group.

        visual:
            How the spin is drawn, or `None` for `default_visual_`.

        color:
            The initial state.

        J:
            The coupling between neighbouring spins.

        temperature:
            The temperature in units of the coupling.

    Attributes:
        default_visual_:
            The visual of the spin when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import PottsSpinElement
        >>> PottsSpinElement('spin').temperature
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        color: Annotated[float, StateSpec(domain=NN)] = 0.0,
        J: Annotated[float, ParameterSpec()] = 1.0,
        temperature: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.color = color
        self.J = J
        self.temperature = temperature


class XYSpinElement(Element):
    """Planar spin of the XY system, an angle `theta`.

    The spin has its coupling `J` and its temperature.

    An `XYSamplingInteraction` rotates the angles by Monte Carlo sweeps at the temperature of the spins.

    Args:
        label:
            The label of the spin or the lattice group.

        visual:
            How the spin is drawn, or `None` for `default_visual_`.

        theta:
            The initial angle.

        J:
            The coupling between neighbouring spins.

        temperature:
            The temperature in units of the coupling.

    Attributes:
        default_visual_:
            The visual of the spin when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import XYSpinElement
        >>> XYSpinElement('spin').temperature
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
        J: Annotated[float, ParameterSpec()] = 1.0,
        temperature: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.J = J
        self.temperature = temperature


class PottsSamplingInteraction(SpinSamplingInteraction):
    """Monte Carlo sweeps of Potts spins in states `0` to `q - 1`.

    The local energy is `-J` times the like neighbours.

    Every move proposes a state drawn from the `q` states. The coupling `J` and the temperature are parameters of the
    spins.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of spins.

        q:
            The number of states.

        state:
            The name of the state that holds the spin.

        rule:
            The acceptance, `'metropolis'`, `'glauber'`, or `'heat-bath'`.

        neighborhood:
            The neighbours of a spin, or `None` for the von Neumann neighbourhood.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics._spins import PottsSamplingInteraction
        >>> PottsSamplingInteraction('potts', target='spins', q=4).values
        (0.0, 1.0, 2.0, 3.0)
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        q: int = 3,
        state: str = 'color',
        rule: str = 'heat-bath',
        neighborhood: BaseNeighborhood | None = None,
    ) -> None:
        super().__init__(
            label,
            target=target,
            state=state,
            energy=self._energy,
            move='random-value',
            rule=rule,
            neighborhood=neighborhood,
            values=tuple(float(value) for value in range(q)),
        )
        self.q = q

    def _energy(self, namespace: SimpleNamespace) -> Expression:
        """Return the local energy `-J` times the number of neighbours in the spin's state."""
        color, other = getattr(namespace.target, self.state), getattr(namespace.target.other, self.state)
        return -namespace.target.J * neighbors(where(other == color, 1, 0), over=self.neighborhood)


class XYSamplingInteraction(SpinSamplingInteraction):
    """Monte Carlo sweeps of planar spins at angles `theta`.

    The local energy is `-J sum_j cos(theta - theta_j)`.

    Every move rotates the angle by a uniform number in `[-width, width]`, accepted by the Metropolis rule. The
    coupling `J` and the temperature are parameters of the spins.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of spins.

        width:
            The largest change of a proposed rotation.

        state:
            The name of the state that holds the angle.

        neighborhood:
            The neighbours of a spin, or `None` for the von Neumann neighbourhood.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics._spins import XYSamplingInteraction
        >>> XYSamplingInteraction('xy', target='spins', width=0.5).move
        'rotate'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        width: float = 1.0,
        state: str = 'theta',
        neighborhood: BaseNeighborhood | None = None,
    ) -> None:
        super().__init__(
            label,
            target=target,
            state=state,
            energy=self._energy,
            move='rotate',
            neighborhood=neighborhood,
            width=width,
        )

    def _energy(self, namespace: SimpleNamespace) -> Expression:
        """Return the local energy `-J sum_j cos(theta - theta_j)` of a spin."""
        angle, other = getattr(namespace.target, self.state), getattr(namespace.target.other, self.state)
        return -namespace.target.J * neighbors(cos(angle - other), over=self.neighborhood)


def _square(size: int | Sequence[int]) -> tuple[int, ...]:
    """Return the shape of a square lattice of a side, or a given shape."""
    if isinstance(size, Sequence):
        return tuple(int(item) for item in size)
    return (int(size), int(size))


class IsingSystem(BaseSystem):
    """The two-dimensional Ising system on a torus.

    The system orders below the critical temperature `2 / ln(1 + sqrt 2)`.

    One unit of time is one sweep of checkerboard updates. The system has the spins `spins.spin`.

    Args:
        size:
            The side of the square lattice, or its shape.

        temperature:
            The temperature in units of the coupling, which can change during a run.

        J:
            The coupling between neighbouring spins.

        h:
            The external field.

        initial:
            `'random'` for random spins or `'up'` for all spins up.

        sampler:
            The acceptance rule, `'metropolis'`, `'glauber'`, or `'heat-bath'`.

        random_state:
            The seed or generator of the random initial spins.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import IsingSystem
        >>> IsingSystem(8, temperature=1.5).prepare().system_.elements[0].values['temperature']
        1.5
    """

    field = 'physics/statistical_mechanics'
    topics = ('lattice', 'Monte Carlo', 'phase transition', 'spins')
    references = ('Ising 1925', 'Onsager 1944', 'Jensen 2023')
    until = 200
    dt = 1

    def __init__(
        self,
        size: int | Sequence[int] = 64,
        *,
        temperature: float = 2.0,
        J: float = 1.0,
        h: float = 0.0,
        initial: str = 'random',
        sampler: str = 'metropolis',
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.temperature = temperature
        self.J = J
        self.h = h
        self.initial = initial
        self.sampler = sampler
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        shape = _square(self.size)
        spins = (
            np.ones(shape)
            if self.initial == 'up'
            else np.random.default_rng(self.random_state).choice([-1.0, 1.0], size=shape)
        )
        space = LatticeSpace(shape)
        group = GroupElement(
            'spins',
            element=IsingSpinElement,
            space=space,
            values={'spin': spins, 'J': self.J, 'h': self.h, 'temperature': self.temperature},
            on='sites',
        )
        return CompositeSystem(
            self.label,
            elements=[group],
            interactions=[IsingSamplingInteraction('sampling', target='spins', rule=self.sampler)],
        )


class PottsSystem(BaseSystem):
    """The q-state Potts system on a torus.

    The system orders below the critical temperature `1 / ln(1 + sqrt q)`.

    The system has the states `spins.color`.

    Args:
        size:
            The side of the square lattice, or its shape.

        q:
            The number of states.

        temperature:
            The temperature in units of the coupling.

        J:
            The coupling between neighbouring spins.

        sampler:
            The acceptance rule, `'metropolis'`, `'glauber'`, or `'heat-bath'`.

        random_state:
            The seed or generator of the random initial states.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import PottsSystem
        >>> PottsSystem(8, q=4).prepare().system_.interactions[0].q
        4
    """

    field = 'physics/statistical_mechanics'
    topics = ('lattice', 'Monte Carlo', 'phase transition', 'spins')
    references = ('Potts 1952', 'Wu 1982')
    until = 200
    dt = 1

    def __init__(
        self,
        size: int | Sequence[int] = 64,
        *,
        q: int = 3,
        temperature: float = 1.0,
        J: float = 1.0,
        sampler: str = 'heat-bath',
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.q = q
        self.temperature = temperature
        self.J = J
        self.sampler = sampler
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the q-state Potts system on a torus.

        The system orders below the critical temperature `1 / ln(1 + sqrt q)`.
        """
        shape = _square(self.size)
        colors = np.random.default_rng(self.random_state).integers(0, self.q, size=shape).astype(float)
        space = LatticeSpace(shape)
        group = GroupElement(
            'spins',
            element=PottsSpinElement,
            space=space,
            values={'color': colors, 'J': self.J, 'temperature': self.temperature},
            on='sites',
        )
        return CompositeSystem(
            self.label,
            elements=[group],
            interactions=[PottsSamplingInteraction('sampling', target='spins', q=self.q, rule=self.sampler)],
        )


class XYSystem(BaseSystem):
    """The XY system of planar spins on a torus.

    Its vortices bind below the Kosterlitz-Thouless point.

    The system has the angles `spins.theta`.

    Args:
        size:
            The side of the square lattice, or its shape.

        temperature:
            The temperature in units of the coupling.

        J:
            The coupling between neighbouring spins.

        width:
            The largest change of a proposed rotation.

        random_state:
            The seed or generator of the random initial angles.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import XYSystem
        >>> XYSystem(8).prepare().system_.interactions[0].width
        1.0
    """

    field = 'physics/statistical_mechanics'
    topics = ('lattice', 'Monte Carlo', 'vortices', 'spins')
    references = ('Kosterlitz and Thouless 1973', 'Jensen 2023')
    until = 200
    dt = 1

    def __init__(
        self,
        size: int | Sequence[int] = 64,
        *,
        temperature: float = 0.5,
        J: float = 1.0,
        width: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.temperature = temperature
        self.J = J
        self.width = width
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        shape = _square(self.size)
        angles = np.random.default_rng(self.random_state).uniform(0, 2 * np.pi, size=shape)
        space = LatticeSpace(shape)
        group = GroupElement(
            'spins',
            element=XYSpinElement,
            space=space,
            values={'theta': angles, 'J': self.J, 'temperature': self.temperature},
            on='sites',
        )
        return CompositeSystem(
            self.label,
            elements=[group],
            interactions=[XYSamplingInteraction('sampling', target='spins', width=self.width)],
        )
