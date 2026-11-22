"""Automata on groups: majority, parity, and linear rules, and a golden mean shift.

The linear rule is the discrete Laplacian. The majority and parity rules run on Cayley balls.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import ZZ

from ....core.components import Observable, SubshiftSpec
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, TotalisticSpec, UpdateLaw
from ....core.components.spaces import BaseNeighborhood, BaseSpace, LatticeNeighborhood, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ..common.automata import CayleySpace, MemoryNeighborhood
from ._binary_cell import BinaryCellElement
from ._generator_neighborhood import GeneratorNeighborhood
from ._groups import baumslag_solitar, free_group, grigorchuk_group, heisenberg, lamplighter

GOLDEN_MEAN = SubshiftSpec(LatticeNeighborhood(((0,), (1,))), ((1, 1),))
GRIGORCHUK_MEMORY = MemoryNeighborhood(('1', 'a', 'b', 'c', 'd'))
LAMPLIGHTER_MEMORY = MemoryNeighborhood(('1', 'a', 'b'))
FLIP_PROBABILITY = 0.5
FREE_GENERATORS = GeneratorNeighborhood(('a', 'b'))
THREE_CELLS = LatticeNeighborhood(((-1,), (0,), (1,)))


class GoldenCellElement(Element):
    """Cell of the golden mean shift, whose state `alive` is 0 or 1.

    Its configurations hold no two adjacent ones. With `space`, the element holds one cell on every site.

    Args:
        label:
            The label of the cells.

        alive:
            The initial value, 0 or 1, or one value per cell.

        space:
            The space with one cell on every site, such as a lattice, or `None` for one cell.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import GoldenCellElement
        >>> GoldenCellElement('cell', alive=1).alive
        1
    """

    def __init__(
        self,
        label: str,
        *,
        alive: Annotated[float | Array, GOLDEN_MEAN] = 0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.alive = alive


class LaplacianCellElement(Element):
    """Cell of an automaton over the field of a prime number of elements.

    An example is the discrete Laplacian on a ring. With `space`, the element holds one cell on every site.

    Args:
        label:
            The label of the cells.

        value:
            The initial value, an integer from 0 to the size of the field minus 1, or one value per cell.

        space:
            The space with one cell on every site, such as a lattice, or `None` for one cell.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LaplacianCellElement
        >>> LaplacianCellElement('cell', value=2).value
        2
    """

    def __init__(
        self,
        label: str,
        *,
        value: Annotated[float | Array, ZZ] = 0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.value = value


class MajorityRuleInteraction(Interaction):
    """Majority rule of binary cells on the sites of a group.

    Every cell takes the value held by more than half of itself and its neighbours, and a tie keeps its value. Only the
    neighbours inside the space count, so the cells on the boundary of a ball read fewer neighbours.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose state `alive` the rule updates.

        neighborhood:
            The neighbours of every cell, such as the generators of the group and their inverses.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import GeneratorNeighborhood, MajorityRuleInteraction
        >>> rule = MajorityRuleInteraction('majority', target='cells', neighborhood=GeneratorNeighborhood(['a', 'b']))
        >>> rule.neighborhood.generators
        ['a', 'b']
    """

    def __init__(self, label: str, *, target: Role = 'cells', neighborhood: BaseNeighborhood = FREE_GENERATORS) -> None:
        super().__init__(label)
        self.target = target
        self.neighborhood = neighborhood

    @staticmethod
    def _get_majority(center: Array, *values: Array, degree: Array, **kwargs: object) -> Array:
        """Return the value held by more than half of a cell and its neighbours, keeping it on a tie."""
        total = center + np.sum(values, axis=0)
        count = degree + 1
        return np.where(2 * total > count, 1.0, np.where(2 * total < count, 0.0, center))

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the majority of every cell and its neighbours as the law.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        alive = context.target.alive
        self.laws_: list[BaseLaw] = [
            UpdateLaw(alive, self._get_majority, neighborhood=self.neighborhood, label=self.label),
        ]


class ParityRuleInteraction(Interaction):
    """Parity rule of binary cells, the sum modulo 2 of a cell and its neighbours.

    It is a linear rule over the field of two elements, so a sum of configurations maps to the sum of their images.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose state `alive` the rule updates.

        neighborhood:
            The cell and its neighbours, such as a memory set of a group.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import ParityRuleInteraction
        >>> from skcomplex.systems.computer_science.common.automata import MemoryNeighborhood
        >>> memory = MemoryNeighborhood(('1', 'a', 'b'))
        >>> ParityRuleInteraction('parity', target='cells', neighborhood=memory).neighborhood.elements
        ('1', 'a', 'b')
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'cells',
        neighborhood: MemoryNeighborhood = LAMPLIGHTER_MEMORY,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.neighborhood = neighborhood

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the parity of every cell and its neighbours as the law.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        parity = TotalisticSpec({total: total % 2 for total in range(len(self.neighborhood.elements) + 1)})
        alive = context.target.alive
        self.laws_: list[BaseLaw] = [
            UpdateLaw(alive, parity, neighborhood=self.neighborhood, label=self.label),
        ]


class LaplacianRuleInteraction(Interaction):
    """Discrete Laplacian `x(g - 1) + x(g + 1) - 2 x(g)` of cells on a ring, modulo the size of a field.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose state `value` the rule updates.

        modulus:
            The prime size of the field.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LaplacianRuleInteraction
        >>> LaplacianRuleInteraction('laplacian', target='cells', modulus=5).modulus
        5
    """

    def __init__(self, label: str, *, target: Role = 'cells', modulus: int = 3) -> None:
        super().__init__(label)
        self.target = target
        self.modulus = modulus

    def _get_function(self) -> Callable[[Array, Array, Array], Array]:
        """Return the discrete Laplacian taken modulo the field size."""
        modulus = self.modulus

        def laplacian(left: Array, center: Array, right: Array, **kwargs: object) -> Array:
            return (left + right - 2 * center) % modulus

        return laplacian

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the Laplacian of every cell and its two neighbours as the law.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        value = context.target.value
        self.laws_: list[BaseLaw] = [
            UpdateLaw(value, self._get_function(), neighborhood=THREE_CELLS, label=self.label),
        ]


class GoldenMeanMoveInteraction(Interaction):
    """Random flips of binary cells that never put two ones side by side.

    Every step proposes to flip every cell, in a random order, with probability one half, and rejects the flips that
    would put a one next to another one.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose state `alive` the flips change.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import GoldenMeanMoveInteraction
        >>> GoldenMeanMoveInteraction('flips', target='cells').target
        'cells'
    """

    def __init__(self, label: str, *, target: Role = 'cells') -> None:
        super().__init__(label)
        self.target = target

    @staticmethod
    def _get_move(left: Array, center: Array, right: Array, *, random: Array, **kwargs: object) -> Array:
        """Return the cell flipped with probability one half, unless a 1 would sit next to another 1."""
        flipped = np.where(random < FLIP_PROBABILITY, 1.0 - center, center)
        blocked = (flipped == 1) & ((left == 1) | (right == 1))
        return np.where(blocked, center, flipped)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the flips, one cell after the other, as the law.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        alive = context.target.alive
        self.laws_: list[BaseLaw] = [
            UpdateLaw(
                alive,
                self._get_move,
                neighborhood=THREE_CELLS,
                schedule='random-sequential',
                label=self.label,
                stochastic=True,
            ),
        ]


class FreeGroupAutomatonSystem(BaseSystem):
    """The majority automaton on a ball of the free group.

    The ball is a tree in which every cell follows its neighbours. Every cell reads itself and the cells at its
    generators and their inverses, and takes the majority value, so the random start freezes into blocks of agreeing
    cells. Cells on the boundary count only their neighbours inside the ball. The system has the cells `cells.alive`.

    Args:
        rank:
            The number of generators.

        radius:
            The radius of the ball.

        density:
            The probability that a cell starts at 1.

        random_state:
            The seed or generator of the initial cells.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import FreeGroupAutomatonSystem
        >>> FreeGroupAutomatonSystem(radius=3).prepare().simulate(until=1).get_values('cells.alive').shape
        (2, 53)
    """

    field = 'computer_science/automata'
    topics = ('group', 'cellular automaton', 'majority', 'free group')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 20
    dt = 1
    observables = (Observable('density', group='cells', variable='alive'),)

    def __init__(
        self,
        rank: int = 2,
        *,
        radius: int = 6,
        density: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.rank = rank
        self.radius = radius
        self.density = density
        self.random_state = random_state

    def _get_cells(self, space: BaseSpace) -> Array:
        """Return cells that are 1 with the probability `density`, one per site of a prepared space."""
        size = space.prepare().size_
        return np.asarray(np.random.default_rng(self.random_state).random(size) < self.density, dtype=float)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of random cells on a ball of the group and their majority rule."""
        super()._prepare(context)
        space = CayleySpace(free_group(self.rank), ball=self.radius)
        neighborhood = GeneratorNeighborhood(space.prepare().generators_)
        self.system_ = System(
            self.label,
            elements=[BinaryCellElement('cells', alive=self._get_cells(space), space=space)],
            interactions=[MajorityRuleInteraction('majority', target='cells', neighborhood=neighborhood)],
        )


class LamplighterAutomatonSystem(BaseSystem):
    """The parity automaton on a ball of the lamplighter group.

    It is a linear rule over the field of two elements. Every cell takes the sum modulo 2 of itself and the cells at the
    generators `a` and `b`, so a few ones spread into nested, self-similar patterns over the Diestel-Leader graph of the
    group. The system has the cells `cells.alive`.

    Args:
        radius:
            The radius of the ball.

        density:
            The probability that a cell starts at 1.

        random_state:
            The seed or generator of the initial cells.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LamplighterAutomatonSystem
        >>> LamplighterAutomatonSystem(radius=3).prepare().simulate(until=1).get_values('cells.alive').shape
        (2, 39)
    """

    field = 'computer_science/automata'
    topics = ('group', 'cellular automaton', 'linear', 'lamplighter')
    references = ('Ceccherini-Silberstein and Coornaert 2023', 'Grigorchuk and Zuk 2001')
    until = 20
    dt = 1
    observables = (Observable('density', group='cells', variable='alive'),)

    def __init__(
        self,
        *,
        radius: int = 7,
        density: float = 0.1,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.radius = radius
        self.density = density
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of random cells on a ball of the lamplighter group and their parity rule."""
        super()._prepare(context)
        space = CayleySpace(lamplighter(), ball=self.radius)
        cells = np.asarray(
            np.random.default_rng(self.random_state).random(space.prepare().size_) < self.density,
            dtype=float,
        )
        self.system_ = System(
            self.label,
            elements=[BinaryCellElement('cells', alive=cells, space=space)],
            interactions=[ParityRuleInteraction('parity', target='cells', neighborhood=LAMPLIGHTER_MEMORY)],
        )


class HeisenbergAutomatonSystem(BaseSystem):
    """The majority automaton on a ball of the discrete Heisenberg group.

    The group has polynomial growth of degree 4. Every cell reads itself and the cells at `a`, `a^-1`, `b`, and `b^-1`,
    on the right, and takes the majority value. The system has the cells `cells.alive`.

    Args:
        radius:
            The radius of the ball.

        density:
            The probability that a cell starts at 1.

        random_state:
            The seed or generator of the initial cells.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import HeisenbergAutomatonSystem
        >>> HeisenbergAutomatonSystem(radius=2).prepare().simulate(until=1).get_values('cells.alive').shape
        (2, 17)
    """

    field = 'computer_science/automata'
    topics = ('group', 'cellular automaton', 'majority', 'nilpotent')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 20
    dt = 1
    observables = (Observable('density', group='cells', variable='alive'),)

    def __init__(
        self,
        *,
        radius: int = 5,
        density: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.radius = radius
        self.density = density
        self.random_state = random_state

    def _get_cells(self, space: BaseSpace) -> Array:
        """Return cells that are 1 with the probability `density`, one per site of a prepared space."""
        size = space.prepare().size_
        return np.asarray(np.random.default_rng(self.random_state).random(size) < self.density, dtype=float)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of random cells on a ball of the group and their majority rule."""
        super()._prepare(context)
        space = CayleySpace(heisenberg(), ball=self.radius)
        neighborhood = GeneratorNeighborhood(space.prepare().generators_)
        self.system_ = System(
            self.label,
            elements=[BinaryCellElement('cells', alive=self._get_cells(space), space=space)],
            interactions=[MajorityRuleInteraction('majority', target='cells', neighborhood=neighborhood)],
        )


class BaumslagSolitarAutomatonSystem(BaseSystem):
    """The majority automaton on a ball of the Baumslag-Solitar group `BS(m, n)`.

    The group has exponential growth. Every cell reads itself and the cells at `a`, `a^-1`, `t`, and `t^-1`, and takes
    the majority value. The ball is built from a confluent rewriting system of the presentation, which GAP finds for
    `BS(1, n)`. The system has the cells `cells.alive`.

    Args:
        m:
            The exponent of `a` conjugated by `t`.

        n:
            The exponent it is conjugated to.

        radius:
            The radius of the ball.

        density:
            The probability that a cell starts at 1.

        random_state:
            The seed or generator of the initial cells.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import BaumslagSolitarAutomatonSystem
        >>> BaumslagSolitarAutomatonSystem(radius=2).prepare().simulate(until=1).get_values('cells.alive').shape
        (2, 17)
    """

    field = 'computer_science/automata'
    topics = ('group', 'cellular automaton', 'majority', 'solvable')
    references = ('Ceccherini-Silberstein and Coornaert 2023', 'Baumslag and Solitar 1962')
    until = 20
    dt = 1
    observables = (Observable('density', group='cells', variable='alive'),)

    def __init__(
        self,
        m: int = 1,
        n: int = 2,
        *,
        radius: int = 6,
        density: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.m = m
        self.n = n
        self.radius = radius
        self.density = density
        self.random_state = random_state

    def _get_cells(self, space: BaseSpace) -> Array:
        """Return cells that are 1 with the probability `density`, one per site of a prepared space."""
        size = space.prepare().size_
        return np.asarray(np.random.default_rng(self.random_state).random(size) < self.density, dtype=float)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of random cells on a ball of the group and their majority rule."""
        super()._prepare(context)
        space = CayleySpace(baumslag_solitar(self.m, self.n), ball=self.radius)
        neighborhood = GeneratorNeighborhood(space.prepare().generators_)
        self.system_ = System(
            self.label,
            elements=[BinaryCellElement('cells', alive=self._get_cells(space), space=space)],
            interactions=[MajorityRuleInteraction('majority', target='cells', neighborhood=neighborhood)],
        )


class LinearRingAutomatonSystem(BaseSystem):
    """The discrete Laplacian as a linear automaton on a ring.

    It runs over the field of `modulus` elements.

    The ring is the cyclic group of order `size`, a periodic lattice, and every cell becomes `x(g - 1) + x(g + 1) - 2
    x(g)` modulo the field size, the element `a^-1 + a - 2` of the group ring. From a single 1 in the middle the values
    spread as binomial coefficients reduced modulo the field size, a self-similar triangle. The system has the cells
    `cells.value`.

    Args:
        size:
            The order of the cyclic group, the number of cells of the ring.

        modulus:
            The prime size of the field.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LinearRingAutomatonSystem
        >>> LinearRingAutomatonSystem(11).prepare().simulate(until=1).get_values('cells.value').shape
        (2, 11)
    """

    field = 'computer_science/automata'
    topics = ('group', 'cellular automaton', 'linear', 'Laplacian')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 40
    dt = 1
    observables = (Observable('mean', group='cells', variable='value'),)

    def __init__(
        self,
        size: int = 101,
        *,
        modulus: int = 3,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.size = size
        self.modulus = modulus

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a ring with a single 1 in the middle and the Laplacian."""
        super()._prepare(context)
        values = np.zeros(self.size)
        values[self.size // 2] = 1.0
        self.system_ = System(
            self.label,
            elements=[LaplacianCellElement('cells', value=values, space=LatticeSpace(self.size))],
            interactions=[LaplacianRuleInteraction('laplacian', target='cells', modulus=self.modulus)],
        )


class SubshiftsSystem(BaseSystem):
    """A Markov chain on the golden mean shift.

    The shift holds the configurations of a ring that never hold two adjacent ones.

    The ring is the cyclic group of order `size`, a periodic lattice. It starts empty, and every step proposes to flip
    every cell in a random order, rejecting the flips that would put two ones side by side. The chain samples the
    admissible configurations, whose number grows as the power of the golden ratio. The system has the cells
    `cells.alive`.

    Args:
        size:
            The number of cells of the ring, the order of the cyclic group.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import SubshiftsSystem
        >>> SubshiftsSystem(10).prepare().simulate(until=1).get_values('cells.alive').shape
        (2, 10)
    """

    field = 'computer_science/automata'
    topics = ('group', 'subshift', 'golden mean', 'Markov chain')
    references = ('Ceccherini-Silberstein and Coornaert 2023', 'Lind and Marcus 1995')
    until = 50
    dt = 1
    observables = (Observable('density', group='cells', variable='alive'),)

    def __init__(
        self,
        size: int = 200,
        *,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.size = size

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of an empty ring of the golden mean shift and its random flips."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[GoldenCellElement('cells', space=LatticeSpace(self.size))],
            interactions=[GoldenMeanMoveInteraction('flips', target='cells')],
        )


class GrigorchukAutomatonSystem(BaseSystem):
    """The majority automaton on a ball of the Grigorchuk group.

    The group has intermediate growth. Every cell reads itself and the cells at the generators `a`, `b`, `c`, and `d`,
    all of order 2, and takes the majority value. The ball of radius 8 holds 271 sites, and cells on its boundary count
    only their neighbours inside it. The system has the cells `cells.alive`.

    Args:
        radius:
            The radius of the ball.

        density:
            The probability that a cell starts at 1.

        random_state:
            The seed or generator of the initial cells.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import GrigorchukAutomatonSystem
        >>> GrigorchukAutomatonSystem(radius=3).prepare().simulate(until=1).get_values('cells.alive').shape
        (2, 23)
    """

    field = 'computer_science/automata'
    topics = ('group', 'cellular automaton', 'majority', 'intermediate growth')
    references = ('Ceccherini-Silberstein and Coornaert 2023', 'Grigorchuk 1984')
    until = 20
    dt = 1
    observables = (Observable('density', group='cells', variable='alive'),)

    def __init__(
        self,
        *,
        radius: int = 8,
        density: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.radius = radius
        self.density = density
        self.random_state = random_state

    def _get_cells(self, space: BaseSpace) -> Array:
        """Return cells that are 1 with the probability `density`, one per site of a prepared space."""
        size = space.prepare().size_
        return np.asarray(np.random.default_rng(self.random_state).random(size) < self.density, dtype=float)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of random cells on a ball of the group and their majority rule."""
        super()._prepare(context)
        space = CayleySpace(grigorchuk_group(), ball=self.radius)
        neighborhood = GRIGORCHUK_MEMORY
        self.system_ = System(
            self.label,
            elements=[BinaryCellElement('cells', alive=self._get_cells(space), space=space)],
            interactions=[MajorityRuleInteraction('majority', target='cells', neighborhood=neighborhood)],
        )
