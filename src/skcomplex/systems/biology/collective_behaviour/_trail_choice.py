"""Choice of a branch by arriving ants, with Deneubourg's probability `(k + n)^h / ((k + n)^h + (k + n')^h)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import ReactionInteraction
from ....core.specs import ParameterSpec, RoleSpec
from ....core.types import Stoichiometry

MODES = {False: 'rate', True: 'jump'}


class TrailChoiceInteraction(ReactionInteraction):
    """Choice of a branch by arriving ants, with Deneubourg's probability `(k + n)^h / ((k + n)^h + (k + n')^h)`.

    Ants arrive at the fork at rate `flux`, and each takes the chosen branch with a probability that grows with its
    marks `n` against the marks `n'` of the other branch, and marks it. It is a `ReactionInteraction` that produces one
    mark on the chosen branch at that propensity and reads the other branch as a modifier, a rate equation, or in
    stochastic mode a random event per ant.

    Args:
        label:
            The label of the interaction.

        chosen:
            The path of the branch this choice marks.

        other:
            The path of the competing branch.

        flux:
            The rate at which ants arrive at the fork.

        k:
            The attractiveness of an unmarked branch.

        h:
            The nonlinearity of the choice.

        stochastic:
            Whether ants are random events.

    Examples:
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.biology.collective_behaviour import BranchElement, TrailChoiceInteraction
        >>> choice = TrailChoiceInteraction('choice', chosen='left', other='right', stochastic=False)
        >>> system = System(elements=[BranchElement('left'), BranchElement('right')], interactions=[choice])
        >>> trajectory = Simulation(system, dt=0.1).prepare().run(until=2.0).trajectory_
        >>> round(float(trajectory.values('left.n')[-1]), 3)
        1.025
    """

    def __init__(
        self,
        label: str,
        *,
        chosen: Annotated[str, RoleSpec()],
        other: Annotated[str, RoleSpec()],
        flux: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        h: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        stochastic: bool = True,
    ) -> None:
        super().__init__(label)
        self.chosen = chosen
        self.other = other
        self.flux = flux
        self.k = k
        self.h = h
        self.stochastic = stochastic

    def _get_products(self) -> Stoichiometry:
        """Return the marks `n` of the chosen branch, which every choice adds one to."""
        return {(self.chosen, 'n'): 1}

    def _get_modifiers(self) -> Stoichiometry:
        """Return the marks `n` of the other branch, which the choice reads."""
        return {(self.other, 'n'): 1}

    def _get_mode(self) -> str:
        """Return the mode of the reaction, random events when stochastic and a rate equation otherwise."""
        return MODES[self.stochastic]

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the rate of ants that choose the branch, `flux` times Deneubourg's probability.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the chosen and the other branch.

        Returns:
            rate:
                The propensity of the marking.
        """
        s = namespace
        attraction = (s.k + s.chosen.n) ** s.h
        propensity: Expression = s.flux * attraction / (attraction + (s.k + s.other.n) ** s.h)
        return propensity
