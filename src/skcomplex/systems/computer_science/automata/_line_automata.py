"""Binary automata on the integers with long memory sets.

They are a nilpotent automaton and an involution.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt

from ....core.components import Observable
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.spaces import LatticeNeighborhood, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ParameterError
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ._binary_cell import BinaryCellElement

MIN_NILPOTENT_CLASS = 2
MIN_INVOLUTION_RUN = 3


class NilpotentRuleInteraction(Interaction):
    """Rule of a binary line that reaches 0 from every configuration in `n` steps.

    Every cell reads the window of offsets `-1, 0, ..., n - 1`. It becomes 0 when the whole window holds ones or when
    it is the left end of a run of ones, and keeps its value otherwise.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose state `alive` the rule updates.

        n:
            The nilpotency class.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import NilpotentRuleInteraction
        >>> NilpotentRuleInteraction('rule', target='cells', n=3).n
        3
    """

    def __init__(self, label: str, *, target: Role = 'cells', n: int = 4) -> None:
        super().__init__(label)
        self.target = target
        self.n = n

    @staticmethod
    def _get_next(*values: Array) -> Array:
        """Return 0 on a window of ones or at the left end of a run of ones, and the cell otherwise.

        The values are those of the offsets `-1, 0, ..., n - 1`, so the cell itself is the second.
        """
        window = np.stack(values)
        left, center = window[0], window[1]
        killed = np.all(window == 1, axis=0) | ((left == 0) & (center == 1))
        return np.where(killed, 0.0, center)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rule on the window of offsets `-1, 0, ..., n - 1` as the law.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        window = LatticeNeighborhood(tuple((offset,) for offset in range(-1, self.n)))
        alive = context.target.alive
        self.laws_: list[BaseLaw] = [UpdateLaw(alive, self._get_next, neighborhood=window, label=self.label)]


class InvolutiveRuleInteraction(Interaction):
    """Rule of a binary line that is its own inverse.

    Every cell reads the window of offsets `-1, 0, ..., n`. The window `alpha = 0 1^n 0` sends the cell to 0, the
    window `beta = 0 0 1^(n - 1) 0` sends it to 1, and every other window keeps it, so two steps return every
    configuration to itself.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose state `alive` the rule updates.

        n:
            The length of the runs of ones.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import InvolutiveRuleInteraction
        >>> InvolutiveRuleInteraction('rule', target='cells', n=4).n
        4
    """

    def __init__(self, label: str, *, target: Role = 'cells', n: int = 3) -> None:
        super().__init__(label)
        self.target = target
        self.n = n

    @staticmethod
    def _get_next(*values: Array) -> Array:
        """Return the cell under the involution.

        The values are those of the offsets `-1, 0, ..., n`, so the cell itself is the second.
        """
        window = np.stack(values)
        run = window.shape[0] - 2
        alpha = np.asarray([0, *([1] * run), 0])[:, None]
        beta = np.asarray([0, 0, *([1] * (run - 1)), 0])[:, None]
        center = window[1]
        return np.where(np.all(window == alpha, axis=0), 0.0, np.where(np.all(window == beta, axis=0), 1.0, center))

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rule on the window of offsets `-1, 0, ..., n` as the law.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        window = LatticeNeighborhood(tuple((offset,) for offset in range(-1, self.n + 1)))
        alive = context.target.alive
        self.laws_: list[BaseLaw] = [UpdateLaw(alive, self._get_next, neighborhood=window, label=self.label)]


class NilpotentAutomatonSystem(BaseSystem):
    """A binary automaton on a ring that is nilpotent of class `n`.

    Every configuration reaches 0 in `n` steps. The memory set is `{-1, 0, ..., n - 1}`, and a cell becomes 0 when the
    whole window holds ones or when it is the left end of a run of ones, `p(-1) = 0` and `p(0) = 1`, and keeps its value
    otherwise (Ceccherini-Silberstein and Coornaert 2023, Exercise 1.15). After one step every run of ones is shorter
    than `n`, and every further step removes the left end of each run, so `n` steps reach 0 and the run of `n + 1` ones
    needs all of them. The exercise prints the memory set `{-1, ..., n - 2}`, whose class is `n - 1`, while its solution
    relies on `{-1, ..., n - 1}`, which this system takes. The system has the cells `cells.alive`.

    Args:
        n:
            The nilpotency class, at least 2.

        size:
            The number of cells of the ring.

        density:
            The probability that a cell starts at 1, when no initial cells are given.

        initial:
            The initial cells, or `None` for random ones.

        random_state:
            The seed or generator of the random initial cells.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ParameterError:
            The class is below 2.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import NilpotentAutomatonSystem
        >>> trajectory = NilpotentAutomatonSystem(3, size=12, density=1.0).prepare().simulate(until=3)
        >>> trajectory.get_values('cells.alive').sum(axis=1).tolist()
        [12, 0, 0, 0]
    """

    field = 'computer_science/automata'
    topics = ('lattice', 'cellular automaton', 'nilpotent')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 6
    dt = 1
    observables = (Observable('density', group='cells', variable='alive'),)

    def __init__(
        self,
        n: int = 4,
        *,
        size: int = 80,
        density: float = 0.7,
        initial: npt.ArrayLike | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n = n
        self.size = size
        self.density = density
        self.initial = initial
        self.random_state = random_state

    def _get_cells(self) -> Array:
        """Return the initial cells, given or 1 with the probability `density`."""
        if self.initial is not None:
            return np.asarray(self.initial, dtype=float)
        return (np.random.default_rng(self.random_state).random(self.size) < self.density).astype(float)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a ring of cells and their nilpotent rule."""
        super()._prepare(context)
        if self.n < MIN_NILPOTENT_CLASS:
            error_msg = f'The nilpotency class of the automaton is at least 2, but it is {self.n}.'
            raise ParameterError(error_msg)
        cells = self._get_cells()
        self.system_ = System(
            self.label,
            elements=[BinaryCellElement('cells', alive=cells, space=LatticeSpace(cells.size))],
            interactions=[NilpotentRuleInteraction('rule', target='cells', n=self.n)],
        )


class InvolutiveAutomatonSystem(BaseSystem):
    """A binary automaton on a ring that is its own inverse.

    Its minimal memory set is `{-1, 0, ..., n}`. A cell at the left end of a run of exactly `n` ones, the window
    `alpha = 0 1^n 0`, becomes 0, a cell just before a run of exactly `n - 1` ones, the window `beta = 0 0 1^(n - 1) 0`,
    becomes 1, and every other cell keeps its value (Ceccherini-Silberstein and Coornaert 2023, Exercise 1.36). The two
    moves undo each other, so two steps return every configuration to itself, and the automaton is invertible with
    `τ^-1 = τ`. The system has the cells `cells.alive`.

    Args:
        n:
            The length of the runs of ones, at least 3.

        size:
            The number of cells of the ring.

        density:
            The probability that a cell starts at 1, when no initial cells are given.

        initial:
            The initial cells, or `None` for random ones.

        random_state:
            The seed or generator of the random initial cells.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ParameterError:
            The run length is below 3.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import InvolutiveAutomatonSystem
        >>> values = InvolutiveAutomatonSystem(3, size=20).prepare().simulate(until=2).get_values('cells.alive')
        >>> bool((values[2] == values[0]).all())
        True
    """

    field = 'computer_science/automata'
    topics = ('lattice', 'cellular automaton', 'invertible', 'involution')
    references = ('Ceccherini-Silberstein and Coornaert 2023',)
    until = 10
    dt = 1
    observables = (Observable('density', group='cells', variable='alive'),)

    def __init__(
        self,
        n: int = 3,
        *,
        size: int = 80,
        density: float = 0.6,
        initial: npt.ArrayLike | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n = n
        self.size = size
        self.density = density
        self.initial = initial
        self.random_state = random_state

    def _get_cells(self) -> Array:
        """Return the initial cells, given or 1 with the probability `density`."""
        if self.initial is not None:
            return np.asarray(self.initial, dtype=float)
        return (np.random.default_rng(self.random_state).random(self.size) < self.density).astype(float)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a ring of cells and their involutive rule."""
        super()._prepare(context)
        if self.n < MIN_INVOLUTION_RUN:
            error_msg = f'The run length of the involutive automaton is at least 3, but it is {self.n}.'
            raise ParameterError(error_msg)
        cells = self._get_cells()
        self.system_ = System(
            self.label,
            elements=[BinaryCellElement('cells', alive=cells, space=LatticeSpace(cells.size))],
            interactions=[InvolutiveRuleInteraction('rule', target='cells', n=self.n)],
        )
