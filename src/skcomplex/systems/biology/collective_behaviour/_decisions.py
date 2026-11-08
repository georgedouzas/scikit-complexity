"""Collective decisions through stigmergy and contagion."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import CompartmentElement
from ....core.components.interactions import ReactionInteraction
from ....core.components.system import BaseSystem, System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Stoichiometry

MODES = {False: 'rate', True: 'jump'}


class BranchElement(CompartmentElement):
    """Branch of a path that ants can take, holding its number `n` of pheromone marks.

    The marks are the ones left by the ants that took the branch.

    Args:
        label:
            The label of the branch.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n:
            The initial number of marks.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import BranchElement
        >>> BranchElement('left', n=5.0).n
        5.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual, n=n)


class RatsElement(CompartmentElement):
    """Rats in one behavioural state, calm or nervous, counted by the state `n`.

    Args:
        label:
            The label of the state.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n:
            The initial number of rats.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import RatsElement
        >>> RatsElement('calm', n=90.0).n
        90.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual, n=n)


class TrailChoiceInteraction(ReactionInteraction):
    """Choice of a branch by arriving ants, with Deneubourg's probability.

    The probability is `(k + n)^h / ((k + n)^h + (k + n')^h)`.

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
        >>> from skcomplex.systems.biology.collective_behaviour import BranchElement, TrailChoiceInteraction
        >>> from skcomplex.core.engine.runtime import simulate
        >>> choice = TrailChoiceInteraction('choice', chosen='left', other='right', stochastic=False)
        >>> system = System(elements=[BranchElement('left'), BranchElement('right')], interactions=[choice])
        >>> trajectory = simulate(system, until=2.0, dt=0.1)
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
        """Return the marks `n` of the chosen branch.

        Every choice adds one to them.
        """
        return {(self.chosen, 'n'): 1}

    def _get_modifiers(self) -> Stoichiometry:
        """Return the marks `n` of the other branch.

        The choice reads them.
        """
        return {(self.other, 'n'): 1}

    def _get_mode(self) -> str:
        """Return the mode of the reaction.

        The mode is random events when stochastic and a rate equation otherwise.
        """
        return MODES[self.stochastic]

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the rate of ants that choose the branch.

        The rate is `flux` times Deneubourg's probability.

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


class AlarmInteraction(ReactionInteraction):
    """Calm rats that turn nervous on their own or by meeting nervous rats.

    They turn nervous at the alarm propensity.

    A calm rat turns nervous at rate `spontaneous + contagion * nervous / N`, where `N` is the total number of rats, so
    the propensity of the reaction calm to nervous is that rate times the calm rats. It is a `ReactionInteraction` that
    moves one rat from the calm to the nervous rats, a rate equation, or in stochastic mode a random event per rat.

    Args:
        label:
            The label of the interaction.

        calm:
            The path of the calm rats.

        nervous:
            The path of the nervous rats.

        spontaneous:
            The rate at which a calm rat turns nervous on its own.

        contagion:
            The rate at which a calm rat turns nervous when all others are nervous.

        stochastic:
            Whether changes are random events.

    Examples:
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.systems.biology.collective_behaviour import AlarmInteraction, RatsElement
        >>> from skcomplex.core.engine.runtime import simulate
        >>> alarm = AlarmInteraction('alarm', calm='calm', nervous='nervous', stochastic=False)
        >>> rats = [RatsElement('calm', n=90.0), RatsElement('nervous', n=10.0)]
        >>> system = System(elements=rats, interactions=[alarm])
        >>> trajectory = simulate(system, until=5.0, dt=0.1)
        >>> round(float(trajectory.values('calm.n')[-1] + trajectory.values('nervous.n')[-1]), 6)
        100.0
    """

    def __init__(
        self,
        label: str,
        *,
        calm: Annotated[str, RoleSpec()],
        nervous: Annotated[str, RoleSpec()],
        spontaneous: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        contagion: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        stochastic: bool = True,
    ) -> None:
        super().__init__(label)
        self.calm = calm
        self.nervous = nervous
        self.spontaneous = spontaneous
        self.contagion = contagion
        self.stochastic = stochastic

    def _get_reactants(self) -> Stoichiometry:
        """Return the calm rats `n`.

        Every alarm takes one from them.
        """
        return {(self.calm, 'n'): 1}

    def _get_products(self) -> Stoichiometry:
        """Return the nervous rats `n`.

        Every alarm adds one to them.
        """
        return {(self.nervous, 'n'): 1}

    def _get_mode(self) -> str:
        """Return the mode of the reaction.

        The mode is random events when stochastic and a rate equation otherwise.
        """
        return MODES[self.stochastic]

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the alarm propensity.

        The propensity is the rate at which a calm rat turns nervous times the calm rats.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the calm and the nervous rats.

        Returns:
            rate:
                The propensity of the alarm.
        """
        s = namespace
        calm, nervous = s.calm.n, s.nervous.n
        propensity: Expression = (s.spontaneous + s.contagion * nervous / (calm + nervous)) * calm
        return propensity


class BinaryAntDecisionSystem(BaseSystem):
    """Ants that choose between two equal branches to a food source.

    The ants follow the marks of earlier ants.

    Every ant reinforces the branch it takes, so the first random fluctuations grow and most ants end up on one branch,
    the symmetry breaking of the double bridge experiment. The deterministic mode stays symmetric. The system has the
    marks `left.n` and `right.n`.

    Args:
        flux:
            The rate at which ants arrive at the fork.

        k:
            The attractiveness of an unmarked branch.

        h:
            The nonlinearity of the choice.

        evaporation:
            The rate at which each mark evaporates.

        stochastic:
            Whether ants and evaporation are random events.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import BinaryAntDecisionSystem
        >>> [branch.label for branch in BinaryAntDecisionSystem().prepare().system_.elements]
        ['left', 'right']
    """

    field = 'biology/collective_behaviour'
    topics = ('stigmergy', 'collective decision', 'symmetry breaking', 'jump process')
    references = ('Deneubourg et al. 1990', 'Gros 2015')
    until = 200.0
    dt = 1.0

    def __init__(
        self,
        *,
        flux: float = 1.0,
        k: float = 20.0,
        h: float = 2.0,
        evaporation: float = 0.0,
        stochastic: bool = True,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.flux = flux
        self.k = k
        self.h = h
        self.evaporation = evaporation
        self.stochastic = stochastic

    def _build_system(self) -> System:
        """Return the parts of ants that choose between two equal branches to food.

        The branches lead to a food source.
        """
        branches = [BranchElement(label) for label in ('left', 'right')]
        evaporations = [
            ReactionInteraction(
                f'evaporation_{label}',
                reactants={(label, 'n'): 1},
                k=self.evaporation,
                mode=MODES[self.stochastic],
            )
            for label in ('left', 'right')
        ]
        choices = [
            TrailChoiceInteraction(
                f'choice_{chosen}',
                chosen=chosen,
                other=other,
                flux=self.flux,
                k=self.k,
                h=self.h,
                stochastic=self.stochastic,
            )
            for chosen, other in (('left', 'right'), ('right', 'left'))
        ]
        return System(self.label, elements=branches, interactions=[*evaporations, *choices])


class RatsSystem(BaseSystem):
    """A colony of calm and nervous rats.

    Nervousness spreads by contact and fades on its own.

    Without spontaneous alarm the colony has a phase transition at `contagion = calming`: below it every rat calms down,
    and above it a share `1 - calming / contagion` of the rats stays nervous. The system has the numbers `calm.n` and
    `nervous.n`.

    Args:
        n_rats:
            The number of rats.

        nervous:
            The initial number of nervous rats.

        spontaneous:
            The rate at which a calm rat turns nervous on its own.

        contagion:
            The rate at which a calm rat turns nervous when all others are nervous.

        calming:
            The rate at which a nervous rat calms down.

        stochastic:
            Whether changes are random events.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import RatsSystem
        >>> [group.n for group in RatsSystem().prepare().system_.elements]
        [95.0, 5.0]
    """

    field = 'biology/collective_behaviour'
    topics = ('contagion', 'phase transition', 'jump process')
    references = ('Gros 2015',)
    until = 20.0
    dt = 0.1

    def __init__(
        self,
        n_rats: int = 100,
        *,
        nervous: int = 5,
        spontaneous: float = 0.01,
        contagion: float = 2.0,
        calming: float = 1.0,
        stochastic: bool = True,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_rats = n_rats
        self.nervous = nervous
        self.spontaneous = spontaneous
        self.contagion = contagion
        self.calming = calming
        self.stochastic = stochastic

    def _build_system(self) -> System:
        """Return the parts of a colony of calm and nervous rats."""
        groups = [
            RatsElement('calm', n=float(self.n_rats - self.nervous)),
            RatsElement('nervous', n=float(self.nervous)),
        ]
        alarm = AlarmInteraction(
            'alarm',
            calm='calm',
            nervous='nervous',
            spontaneous=self.spontaneous,
            contagion=self.contagion,
            stochastic=self.stochastic,
        )
        calming_reaction = ReactionInteraction(
            'calming',
            reactants={('nervous', 'n'): 1},
            products={('calm', 'n'): 1},
            k=self.calming,
            mode=MODES[self.stochastic],
        )
        return System(self.label, elements=groups, interactions=[alarm, calming_reaction])
