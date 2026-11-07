"""Statistical mechanics systems: gases, fluids, and spin systems."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, Expression, cos, oo, pi
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, LatticeElement
from ....core.components.laws.utils import neighbors, where
from ....core.components.spaces import BaseNeighborhood, LatticeSpace
from ....core.components.system import System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ..common.spins import IsingSamplingInteraction, IsingSpinElement, SpinSamplingInteraction


class PottsSpinElement(Element):
    """Spin of the q-state Potts system, in a state from 0 to `q - 1`, with its coupling `J` and its temperature.

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
    """Planar spin of the XY system, an angle `theta`, with its coupling `J` and its temperature.

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
    """Monte Carlo sweeps of Potts spins in states `0` to `q - 1`, of local energy `-J` times the like neighbours.

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
        """Return the local energy `-J` times the number of neighbours in the state of the spin."""
        color, other = getattr(namespace.target, self.state), getattr(namespace.target.other, self.state)
        return -namespace.target.J * neighbors(where(other == color, 1, 0), over=self.neighborhood)


class XYSamplingInteraction(SpinSamplingInteraction):
    """Monte Carlo sweeps of planar spins at angles `theta`, of local energy `-J sum_j cos(theta - theta_j)`.

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


@register_system(
    field='physics/statistical_mechanics',
    tags=('lattice', 'Monte Carlo', 'phase transition', 'spins'),
    references=('Ising 1925', 'Onsager 1944', 'Jensen 2023'),
    defaults={'until': 200, 'dt': 1},
)
def ising(
    size: int | Sequence[int] = 64,
    *,
    temperature: float = 2.0,
    J: float = 1.0,
    h: float = 0.0,
    initial: str = 'random',
    sampler: str = 'metropolis',
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the two-dimensional Ising system on a torus, ordered below the critical temperature `2 / ln(1 + sqrt 2)`.

    One unit of time is one sweep of checkerboard updates.

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

    Returns:
        system:
            The system, with the spins `spins.spin`.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ising
        >>> ising(8, temperature=1.5).elements[0].values['temperature']
        1.5
    """
    shape = _square(size)
    spins = np.ones(shape) if initial == 'up' else np.random.default_rng(random_state).choice([-1.0, 1.0], size=shape)
    space = LatticeSpace(shape)
    group = LatticeElement(
        'spins',
        element=IsingSpinElement,
        space=space,
        values={'spin': spins, 'J': J, 'h': h, 'temperature': temperature},
    )
    return System(elements=[group], interactions=[IsingSamplingInteraction('sampling', target='spins', rule=sampler)])


@register_system(
    field='physics/statistical_mechanics',
    tags=('lattice', 'Monte Carlo', 'phase transition', 'spins'),
    references=('Potts 1952', 'Wu 1982'),
    defaults={'until': 200, 'dt': 1},
)
def potts(
    size: int | Sequence[int] = 64,
    *,
    q: int = 3,
    temperature: float = 1.0,
    J: float = 1.0,
    sampler: str = 'heat-bath',
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the q-state Potts system on a torus, ordered below the critical temperature `1 / ln(1 + sqrt q)`.

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

    Returns:
        system:
            The system, with the states `spins.color`.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import potts
        >>> potts(8, q=4).interactions[0].q
        4
    """
    shape = _square(size)
    colors = np.random.default_rng(random_state).integers(0, q, size=shape).astype(float)
    space = LatticeSpace(shape)
    group = LatticeElement(
        'spins',
        element=PottsSpinElement,
        space=space,
        values={'color': colors, 'J': J, 'temperature': temperature},
    )
    return System(
        elements=[group],
        interactions=[PottsSamplingInteraction('sampling', target='spins', q=q, rule=sampler)],
    )


@register_system(
    field='physics/statistical_mechanics',
    tags=('lattice', 'Monte Carlo', 'vortices', 'spins'),
    references=('Kosterlitz and Thouless 1973', 'Jensen 2023'),
    defaults={'until': 200, 'dt': 1},
)
def xy(
    size: int | Sequence[int] = 64,
    *,
    temperature: float = 0.5,
    J: float = 1.0,
    width: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the XY system of planar spins on a torus, with vortices that bind below the Kosterlitz-Thouless point.

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

    Returns:
        system:
            The system, with the angles `spins.theta`.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import xy
        >>> xy(8).interactions[0].width
        1.0
    """
    shape = _square(size)
    angles = np.random.default_rng(random_state).uniform(0, 2 * np.pi, size=shape)
    space = LatticeSpace(shape)
    group = LatticeElement(
        'spins',
        element=XYSpinElement,
        space=space,
        values={'theta': angles, 'J': J, 'temperature': temperature},
    )
    return System(elements=[group], interactions=[XYSamplingInteraction('sampling', target='spins', width=width)])
