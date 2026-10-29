"""Collective behaviour models: collective decisions through stigmergy and contagion."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, JumpLaw, RateLaw
from ....core.system import System
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model


def _transfer(source: Expression, destination: Expression, rate: Expression, *, stochastic: bool) -> list[BaseLaw]:
    """Return the laws that move individuals from one state to another at a rate."""
    if stochastic:
        return [JumpLaw(rate, {source: -1, destination: 1})]
    return [RateLaw(source, -rate), RateLaw(destination, rate)]


class BranchElement(Element):
    """Branch of a path that ants can take, holding the pheromone marks left by the ants that took it.

    Args:
        label:
            The label of the branch.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n:
            The initial number of marks.

        evaporation:
            The rate at which each mark evaporates.

        stochastic:
            Whether marks evaporate one at a time at random.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import BranchElement
        >>> BranchElement('left', n=5.0).n
        5.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        evaporation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        stochastic: bool = True,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
        self.evaporation = evaporation
        self.stochastic = stochastic

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the evaporation of the marks.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        if self.stochastic:
            return [JumpLaw(s.evaporation * s.n, {s.n: -1})]
        return [RateLaw(s.n, -s.evaporation * s.n)]


class TrailChoiceInteraction(Interaction):
    """Choice of a branch by arriving ants, with Deneubourg's probability `(k + n)^h / ((k + n)^h + (k + n')^h)`.

    Ants arrive at the fork at rate `flux`, and each takes the chosen branch with a probability that grows with its
    marks `n` against the marks `n'` of the other branch, and marks it. In stochastic mode every ant is a random event.

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
        >>> from skcomplex.models.biology.collective_behaviour import TrailChoiceInteraction
        >>> TrailChoiceInteraction('left', chosen='left', other='right').h
        2.0
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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the marking of the chosen branch at the rate of ants that choose it.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        attraction = (s.k + s.chosen.n) ** s.h
        rate = s.flux * attraction / (attraction + (s.k + s.other.n) ** s.h)
        if self.stochastic:
            return [JumpLaw(rate, {s.chosen.n: 1})]
        return [RateLaw(s.chosen.n, rate)]


class RatsElement(Element):
    """Rats in one behavioural state, calm or nervous, counted.

    Args:
        label:
            The label of the state.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n:
            The initial number of rats.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import RatsElement
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
        super().__init__(label, visual=visual)
        self.n = n


class NervousnessInteraction(Interaction):
    """Calm rats that become nervous on their own or by meeting nervous rats, and nervous rats that calm down.

    A calm rat turns nervous at rate `spontaneous + contagion * nervous / N` and a nervous one calms down at rate
    `calming`, where `N` is the total number of rats. In stochastic mode every change is a random event.

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

        calming:
            The rate at which a nervous rat calms down.

        stochastic:
            Whether changes are random events.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import NervousnessInteraction
        >>> NervousnessInteraction('panic', calm='calm', nervous='nervous').calming
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        calm: Annotated[str, RoleSpec()],
        nervous: Annotated[str, RoleSpec()],
        spontaneous: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        contagion: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        calming: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        stochastic: bool = True,
    ) -> None:
        super().__init__(label)
        self.calm = calm
        self.nervous = nervous
        self.spontaneous = spontaneous
        self.contagion = contagion
        self.calming = calming
        self.stochastic = stochastic

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the transitions between calm and nervous.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        calm, nervous = s.calm.n, s.nervous.n
        alarm = (s.spontaneous + s.contagion * nervous / (calm + nervous)) * calm
        stochastic = self.stochastic
        return [
            *_transfer(calm, nervous, alarm, stochastic=stochastic),
            *_transfer(nervous, calm, s.calming * nervous, stochastic=stochastic),
        ]


@register_model(
    field='biology/collective_behaviour',
    tags=('stigmergy', 'collective decision', 'symmetry breaking', 'jump process'),
    references=('Deneubourg et al. 1990', 'Gros 2015'),
    defaults={'until': 200.0, 'dt': 1.0},
)
def binary_ant_decision(
    *,
    flux: float = 1.0,
    k: float = 20.0,
    h: float = 2.0,
    evaporation: float = 0.0,
    stochastic: bool = True,
) -> System:
    """Return ants that choose between two equal branches to a food source, following the marks of earlier ants.

    Every ant reinforces the branch it takes, so the first random fluctuations grow and most ants end up on one
    branch, the symmetry breaking of the double bridge experiment. The deterministic mode stays symmetric.

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

    Returns:
        system:
            The model, with the marks `left.n` and `right.n`.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import binary_ant_decision
        >>> [branch.label for branch in binary_ant_decision().elements]
        ['left', 'right']
    """
    branches = [BranchElement(label, evaporation=evaporation, stochastic=stochastic) for label in ('left', 'right')]
    choices = [
        TrailChoiceInteraction(
            f'choice_{chosen}',
            chosen=chosen,
            other=other,
            flux=flux,
            k=k,
            h=h,
            stochastic=stochastic,
        )
        for chosen, other in (('left', 'right'), ('right', 'left'))
    ]
    return System(elements=branches, interactions=choices)


@register_model(
    field='biology/collective_behaviour',
    tags=('contagion', 'phase transition', 'jump process'),
    references=('Gros 2015',),
    defaults={'until': 20.0, 'dt': 0.1},
)
def rats(
    n_rats: int = 100,
    *,
    nervous: int = 5,
    spontaneous: float = 0.01,
    contagion: float = 2.0,
    calming: float = 1.0,
    stochastic: bool = True,
) -> System:
    """Return a colony of calm and nervous rats, in which nervousness spreads by contact and fades on its own.

    Without spontaneous alarm the colony has a phase transition at `contagion = calming`: below it every rat calms
    down, and above it a share `1 - calming / contagion` of the rats stays nervous.

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

    Returns:
        system:
            The model, with the numbers `calm.n` and `nervous.n`.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import rats
        >>> [group.n for group in rats().elements]
        [95.0, 5.0]
    """
    groups = [RatsElement('calm', n=float(n_rats - nervous)), RatsElement('nervous', n=float(nervous))]
    transitions = NervousnessInteraction(
        'nervousness',
        calm='calm',
        nervous='nervous',
        spontaneous=spontaneous,
        contagion=contagion,
        calming=calming,
        stochastic=stochastic,
    )
    return System(elements=groups, interactions=[transitions])
