"""Statistical mechanics systems.

The module covers gases, fluids, and spin systems.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, cos, oo
from sage.sets.real_set import RealSet

from ....core import systems
from ....core.components import Observable
from ....core.components.elements import Element, Period
from ....core.components.laws.utils import neighbors, where
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, MetropolisSolver
from ....core.specs import Role
from ..common.spins import IsingSamplingInteraction, IsingSpinElement, SpinSamplingInteraction


class PottsSpinElement(Element):
    """Spins of the q-state Potts system, each in a state from 0 to `q - 1`.

    Every spin has its coupling `J` and its temperature.

    A `PottsSamplingInteraction` redraws the states by Monte Carlo sweeps at the temperature of the spins.

    Args:
        label:
            The label of the spin or the lattice of spins.

        color:
            The initial state, or one state per site.

        J:
            The coupling between neighbouring spins.

        temperature:
            The temperature in units of the coupling.

        space:
            The lattice of the spins, or `None` for one spin.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import PottsSpinElement
        >>> PottsSpinElement('spin').temperature
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        color: object = 0.0,
        J: float = 1.0,
        temperature: Annotated[float, RealSet.open(0, oo)] = 1.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
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

        theta:
            The initial angle.

        J:
            The coupling between neighbouring spins.

        temperature:
            The temperature in units of the coupling.

        space:
            The lattice of the spins, or `None` for one spin.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import XYSpinElement
        >>> XYSpinElement('spin').temperature
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        theta: Annotated[object, Period(2 * math.pi)] = 0.0,
        J: float = 1.0,
        temperature: Annotated[float, RealSet.open(0, oo)] = 0.5,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.theta = theta
        self.J = J
        self.temperature = temperature


class PottsSamplingInteraction(SpinSamplingInteraction):
    """Boltzmann weight of Potts spins in states `0` to `q - 1`, whose local energy is `-J` times the like neighbours.

    The coupling `J` and the temperature are values of the spins, and the `MetropolisSolver` of the system proposes
    states drawn from the `q` states, `MetropolisSolver(moves={'spins.color': 'random-value'},
    values={'spins.color': range(q)})`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of spins.

        q:
            The number of states.

        state:
            The name of the state that holds the spin.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics._spins import PottsSamplingInteraction
        >>> PottsSamplingInteraction('potts', target='spins', q=4).q
        4
    """

    def __init__(self, label: str, *, target: Role, q: int = 3, state: str = 'color') -> None:
        super().__init__(label, target=target, state=state, energy=self._get_energy)
        self.q = q

    def _get_energy(self, namespace: SimpleNamespace) -> Expression:
        """Return the local energy `-J` times the number of neighbours in the spin's state."""
        color, other = getattr(namespace.target, self.state), getattr(namespace.target.other, self.state)
        return -namespace.target.J * neighbors(where(other == color, 1, 0))


class XYSamplingInteraction(SpinSamplingInteraction):
    """Boltzmann weight of planar spins at angles `theta`, whose local energy is `-J sum_j cos(theta - theta_j)`.

    The coupling `J` and the temperature are values of the spins, and the `MetropolisSolver` of the system rotates the
    angles by a uniform number in `[-width, width]`, `MetropolisSolver(moves={'spins.theta': 'rotate'}, width=1.0)`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of spins.

        state:
            The name of the state that holds the angle.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics._spins import XYSamplingInteraction
        >>> XYSamplingInteraction('xy', target='spins').state
        'theta'
    """

    def __init__(self, label: str, *, target: Role, state: str = 'theta') -> None:
        super().__init__(label, target=target, state=state, energy=self._get_energy)

    def _get_energy(self, namespace: SimpleNamespace) -> Expression:
        """Return the local energy `-J sum_j cos(theta - theta_j)` of a spin."""
        angle, other = getattr(namespace.target, self.state), getattr(namespace.target.other, self.state)
        return -namespace.target.J * neighbors(cos(angle - other))


class IsingSystem(systems.BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import IsingSystem
        >>> IsingSystem(8, temperature=1.5).prepare().system_.elements[0].temperature
        1.5
    """

    field = 'physics/statistical_mechanics'
    topics = ('lattice', 'Monte Carlo', 'phase transition', 'spins')
    references = ('Ising 1925', 'Onsager 1944', 'Jensen 2023')
    until = 200
    dt = 1

    @staticmethod
    def _get_square(size: int | Sequence[int]) -> tuple[int, ...]:
        """Return the shape of a square lattice of a side, or a given shape."""
        if isinstance(size, Sequence):
            return tuple(int(item) for item in size)
        return (int(size), int(size))

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.size = size
        self.temperature = temperature
        self.J = J
        self.h = h
        self.initial = initial
        self.sampler = sampler
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the Ising spins of a torus, all up or random at the start.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        shape = IsingSystem._get_square(self.size)
        spins = (
            np.ones(shape)
            if self.initial == 'up'
            else np.random.default_rng(self.random_state).choice([-1.0, 1.0], size=shape)
        )
        group = IsingSpinElement(
            'spins',
            spin=spins.ravel(),
            J=self.J,
            h=self.h,
            temperature=self.temperature,
            space=LatticeSpace(shape),
        )
        self.system_ = systems.System(
            self.label,
            elements=[group],
            interactions=[IsingSamplingInteraction('sampling', target='spins')],
            solver=MetropolisSolver(moves={'spins.spin': 'flip'}, rule=self.sampler),
        )


class PottsSystem(systems.BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.size = size
        self.q = q
        self.temperature = temperature
        self.J = J
        self.sampler = sampler
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the q-state Potts system on a torus, which orders below the temperature `1 / ln(1 + sqrt q)`.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        shape = IsingSystem._get_square(self.size)
        colors = np.random.default_rng(self.random_state).integers(0, self.q, size=shape).astype(float)
        group = PottsSpinElement(
            'spins',
            color=colors.ravel(),
            J=self.J,
            temperature=self.temperature,
            space=LatticeSpace(shape),
        )
        self.system_ = systems.System(
            self.label,
            elements=[group],
            interactions=[PottsSamplingInteraction('sampling', target='spins', q=self.q)],
            solver=MetropolisSolver(
                moves={'spins.color': 'random-value'},
                rule=self.sampler,
                values={'spins.color': tuple(float(value) for value in range(self.q))},
            ),
        )


class XYSystem(systems.BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import XYSystem
        >>> XYSystem(8).prepare().system_.solver.width
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.size = size
        self.temperature = temperature
        self.J = J
        self.width = width
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the planar spins of a torus at random angles.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        shape = IsingSystem._get_square(self.size)
        angles = np.random.default_rng(self.random_state).uniform(0, 2 * np.pi, size=shape)
        group = XYSpinElement(
            'spins',
            theta=angles.ravel(),
            J=self.J,
            temperature=self.temperature,
            space=LatticeSpace(shape),
        )
        self.system_ = systems.System(
            self.label,
            elements=[group],
            interactions=[XYSamplingInteraction('sampling', target='spins')],
            solver=MetropolisSolver(moves={'spins.theta': 'rotate'}, width=self.width),
        )
