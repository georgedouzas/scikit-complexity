"""Define satisfiability interactions, which reveal constraints on Boolean states and decide them incrementally."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar, cast

from pysat.card import CardEnc
from pysat.formula import IDPool
from pysat.solvers import Solver

from ...errors import ValidationError
from ...specs import CardinalitySpec, ClauseSpec, RoleSpec
from ...types import ElementKind
from ..laws import BaseLaw, RuleLaw
from ..types import ModelViewProtocol
from ._interaction import Interaction


class SatisfiabilityInteraction(Interaction):
    """Reveal constraints on the Boolean states of a group step by step, and decide them with one incremental solver.

    Every step adds the next `reveal` constraints to one PySAT solver, which keeps what it learned, and sets the states
    of the members to a satisfying assignment when one exists. The interaction reports `satisfiable`, 1 or 0, and
    `revealed`, the number of active constraints, every step. A state that is not Boolean is reported by the compiler
    with the code `law-domain`.

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
            A constraint is not a spec or the number revealed per step is not positive, raised by `check`.

    Examples:
        >>> from skcomplex.core.interactions import SatisfiabilityInteraction
        >>> from skcomplex.specs import ClauseSpec
        >>> interaction = SatisfiabilityInteraction('sat', target='bits', state='on', constraints=[ClauseSpec((1, -2))])
        >>> interaction.check().reveal
        1
    """

    reports: ClassVar[tuple[str, ...]] = ('satisfiable', 'revealed')

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RuleLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.GROUP, ElementKind.VARIABLE_GROUP), states=('state',))],
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

    def _check(self) -> None:
        """Raise an error for a constraint that is not a spec, or a number revealed per step that is not positive.

        Raises:
            ValidationError:
                A constraint is not a spec or the number revealed per step is not positive.
        """
        super()._check()
        if self.reveal < 1:
            error_msg = f'The satisfiability {self.label!r} has reveal={self.reveal!r}. Reveal at least 1 per step.'
            raise ValidationError(error_msg)
        for constraint in self.constraints:
            if not isinstance(constraint, ClauseSpec | CardinalitySpec):
                error_msg = f'The constraint {constraint!r} of {self.label!r} is no ClauseSpec or CardinalitySpec.'
                raise ValidationError(error_msg)
            constraint.check()

    def _add(self, constraint: ClauseSpec | CardinalitySpec, solver: Solver, pool: IDPool) -> None:
        """Add one constraint to the solver, encoding a cardinality constraint with variables of the shared pool."""
        if isinstance(constraint, ClauseSpec):
            solver.add_clause(list(constraint.literals))
            return
        encode = CardEnc.atleast if constraint.sense == 'at-least' else CardEnc.atmost
        for clause in encode(lits=list(constraint.literals), bound=constraint.bound, vpool=pool).clauses:
            solver.add_clause(clause)

    def _decide(self, model: ModelViewProtocol) -> None:
        """Reveal the next constraints, decide them, and set the members to a satisfying assignment when one exists."""
        agents = model.agents(self._get_role('target'))
        if 'solver' not in model.storage:
            model.storage.update(
                solver=Solver(name=self.solver),
                pool=IDPool(start_from=len(agents) + 1),
                revealed=0,
            )
        solver = cast(Solver, model.storage['solver'])
        pool = cast(IDPool, model.storage['pool'])
        revealed = cast(int, model.storage['revealed'])
        following = min(revealed + self.reveal, len(self.constraints))
        for constraint in self.constraints[revealed:following]:
            self._add(constraint, solver, pool)
        model.storage['revealed'] = following
        satisfiable = solver.solve()
        if satisfiable:
            holding = {literal for literal in solver.get_model() if literal > 0}
            for index, agent in enumerate(agents, start=1):
                setattr(agent, self.state, float(index in holding))
        model.report('satisfiable', float(satisfiable))
        model.report('revealed', float(following))

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
