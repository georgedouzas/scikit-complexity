"""The binary perceptron as a constraint satisfaction problem.

Its satisfiability collapses at a critical load.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated, ClassVar, cast

import numpy as np
from pysat.card import CardEnc
from pysat.formula import IDPool
from pysat.solvers import Solver
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.spaces import BaseSpace, DiscreteTime
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import SystemViewProtocol
from ....core.errors import ValidationError
from ....core.specs import BaseSpec, RoleSpec, StateSpec
from ....core.types import ElementKind

SENSES = ('at-least', 'at-most')


@dataclass(frozen=True)
class ClauseSpec(BaseSpec):
    """Disjunction of literals.

    A literal is a member index from 1, negative for its negation.

    Args:
        literals:
            The literals of the clause.

    Raises:
        ValidationError:
            A literal is 0 or the clause holds none, raised when its owner checks it.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import ClauseSpec
        >>> ClauseSpec((1, -2)).literals
        (1, -2)
    """

    literals: Sequence[int]

    def _check(self) -> None:
        """Raise an error for a clause without literals or with a literal 0.

        Raises:
            ValidationError:
                A literal is 0 or the clause holds none.
        """
        if not len(self.literals) or 0 in self.literals:
            error_msg = f'The clause {tuple(self.literals)} needs literals, member indices from 1, none of them 0.'
            raise ValidationError(error_msg)


@dataclass(frozen=True)
class CardinalitySpec(BaseSpec):
    """Constraint that at least or at most a bound of its literals hold.

    A literal is a member index from 1, negative for its negation.

    Args:
        literals:
            The literals of the constraint.

        bound:
            The number of literals that bounds those that hold.

        sense:
            `'at-least'` or `'at-most'`.

    Raises:
        ValidationError:
            A literal is 0, the bound lies outside the number of literals, or the sense is unknown, raised when its
            owner checks it.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import CardinalitySpec
        >>> CardinalitySpec((1, -2, 3), 2).sense
        'at-least'
    """

    literals: Sequence[int]
    bound: int
    sense: str = 'at-least'

    def _check(self) -> None:
        """Raise an error for a literal 0, a bound out of range, or an unknown sense.

        Raises:
            ValidationError:
                A literal is 0, the bound lies outside the number of literals, or the sense is unknown.
        """
        if self.sense not in SENSES or 0 in self.literals or not 0 <= self.bound <= len(self.literals):
            error_msg = (
                f'The cardinality constraint of {tuple(self.literals)} has bound {self.bound} and sense '
                f'{self.sense!r}. Use literals that are not 0, a bound up to their number, and a sense of '
                f'{", ".join(SENSES)}.'
            )
            raise ValidationError(error_msg)


class SatisfiabilityInteraction(Interaction):
    """Reveal constraints on the Boolean states of a group and decide them.

    The constraints come step by step, and one incremental solver decides them. Every step adds the next `reveal`
    constraints to one PySAT solver, which keeps what it learned, and sets the states of the members to a satisfying
    assignment when one exists. The interaction reports `satisfiable`, 1 or 0, and `revealed`, the number of active
    constraints, every step. Its role accepts Boolean states only, so a real state raises `RoleElementError`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group.

        state:
            The name of the Boolean state of the members.

        constraints:
            The constraints, in the order they are revealed.

        reveal:
            The number of constraints revealed per step.

        solver:
            The name of the PySAT solver.

        phase:
            The name of the phase of the step the rule runs in, or `None` for the default phase.

    Raises:
        ValidationError:
            A constraint is not a spec or the number revealed per step is not positive, raised by `prepare`.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import SatisfiabilityInteraction
        >>> from skcomplex.systems.computer_science.neural_networks import ClauseSpec
        >>> interaction = SatisfiabilityInteraction('sat', target='bits', state='on', constraints=[ClauseSpec((1, -2))])
        >>> interaction.reveal
        1
    """

    reports: ClassVar[tuple[str, ...]] = ('satisfiable', 'revealed')

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RuleLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[
            str,
            RoleSpec(elements=(ElementKind.GROUP, ElementKind.VARIABLE_GROUP), states=('state',), domains=('boolean',)),
        ],
        state: str,
        constraints: Sequence[ClauseSpec | CardinalitySpec],
        reveal: int = 1,
        solver: str = 'cadical153',
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.constraints = constraints
        self.reveal = reveal
        self.solver = solver
        self.phase = phase

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for a constraint not a spec, or a bad number revealed per step.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                A constraint is not a spec or the number revealed per step is not positive.
        """
        super()._prepare(context)
        if self.reveal < 1:
            error_msg = f'The satisfiability {self.label!r} has reveal={self.reveal!r}. Reveal at least 1 per step.'
            raise ValidationError(error_msg)
        for constraint in self.constraints:
            if not isinstance(constraint, ClauseSpec | CardinalitySpec):
                error_msg = f'The constraint {constraint!r} of {self.label!r} is no ClauseSpec or CardinalitySpec.'
                raise ValidationError(error_msg)
            constraint._check()

    def _add(self, constraint: ClauseSpec | CardinalitySpec, solver: Solver, pool: IDPool) -> None:
        """Add one constraint to the solver.

        A cardinality constraint is encoded with variables of the shared pool.
        """
        if isinstance(constraint, ClauseSpec):
            solver.add_clause(list(constraint.literals))
            return
        encode = CardEnc.atleast if constraint.sense == 'at-least' else CardEnc.atmost
        for clause in encode(lits=list(constraint.literals), bound=constraint.bound, vpool=pool).clauses:
            solver.add_clause(clause)

    def _decide(self, system: SystemViewProtocol) -> None:
        """Reveal the next constraints, decide them, and set the members to a solution.

        The solution is a satisfying assignment, set when one exists.
        """
        agents = system.agents(self._get_role('target'))
        if 'solver' not in system.storage:
            system.storage.update(
                solver=Solver(name=self.solver),
                pool=IDPool(start_from=len(agents) + 1),
                revealed=0,
            )
        solver = cast(Solver, system.storage['solver'])
        pool = cast(IDPool, system.storage['pool'])
        revealed = cast(int, system.storage['revealed'])
        following = min(revealed + self.reveal, len(self.constraints))
        for constraint in self.constraints[revealed:following]:
            self._add(constraint, solver, pool)
        system.storage['revealed'] = following
        satisfiable = solver.solve()
        if satisfiable:
            holding = {literal for literal in solver.get_model() if literal > 0}
            for index, agent in enumerate(agents, start=1):
                setattr(agent, self.state, float(index in holding))
        system.report('satisfiable', float(satisfiable))
        system.report('revealed', float(following))

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of one step.

        Args:
            namespace:
                The namespace of the symbols the laws can use, which the rule does not read.

        Returns:
            laws:
                The rule that reveals and decides constraints.
        """
        return [RuleLaw(self._decide, phase=self.phase)]


class BinaryWeightElement(Element):
    """Weight of a binary perceptron.

    The state 1 stands for `+1` and the state 0 for `-1`.

    Args:
        label:
            The label of the weight or the group.

        on:
            Whether the weight is `+1`.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import BinaryWeightElement
        >>> BinaryWeightElement('weight', on=1.0).on
        1.0
    """

    def __init__(self, label: str, on: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0) -> None:
        super().__init__(label)
        self.on = on


class PerceptronSatisfiabilitySystem(BaseSystem):
    """A binary perceptron that must store random patterns.

    The patterns are revealed one per step, as cardinality constraints. A pattern `xi` of output `sigma` holds when
    `sigma sum_i w_i xi_i > 0`, so with an odd number `N` of weights `w_i = +-1`, when at least `(N + 1) / 2` of the
    literals `w_i = sigma xi_i` hold. The patterns are satisfiable up to the critical load `p / N` of about 0.83 in the
    limit of many weights (Bertin 2021 §6.3.3, Ex. 6.4). The patterns are drawn from `random_state`, so a disorder
    average sweeps it with `sweep`. The system has the group `weights` and the reports `decide.satisfiable` and
    `decide.revealed`.

    Args:
        n_weights:
            The number of weights, which is odd.

        load:
            The number of patterns over the number of weights.

        random_state:
            The seed or generator of the patterns.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The number of weights is even, so a pattern can tie.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import PerceptronSatisfiabilitySystem
        >>> PerceptronSatisfiabilitySystem(3, load=1.0).prepare().system_.interactions[0].constraints[0].bound
        2
    """

    field = 'computer_science/neural_networks'
    topics = ('perceptron', 'satisfiability', 'disorder', 'critical load')
    references = ('Bertin 2021', 'Krauth and Mezard 1989')
    until = 31

    def __init__(
        self,
        n_weights: int = 21,
        *,
        load: float = 1.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_weights = n_weights
        self.load = load
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        if self.n_weights % 2 == 0:
            error_msg = (
                f'perceptron_satisfiability has n_weights={self.n_weights!r}. Use an odd number, so no pattern ties.'
            )
            raise ValidationError(error_msg)
        rng = np.random.default_rng(self.random_state)
        n_patterns = round(self.load * self.n_weights)
        patterns = rng.choice([-1, 1], size=(n_patterns, self.n_weights)) * rng.choice([-1, 1], size=(n_patterns, 1))
        indices = np.arange(1, self.n_weights + 1)
        constraints = [
            CardinalitySpec(tuple((indices * pattern).tolist()), (self.n_weights + 1) // 2, 'at-least')
            for pattern in patterns
        ]
        weights = GroupElement('weights', element=BinaryWeightElement, size=self.n_weights)
        decide = SatisfiabilityInteraction('decide', target='weights', state='on', constraints=constraints)
        return CompositeSystem(self.label, elements=[weights], interactions=[decide], time=DiscreteTime())
