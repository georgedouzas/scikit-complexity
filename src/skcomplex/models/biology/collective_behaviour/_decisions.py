"""Collective behaviour models: collective decisions through stigmergy and contagion."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.interactions import ReactionInteraction
from ....core.system import System
from ..._catalog import register_model
from ._alarm import AlarmInteraction
from ._branch import BranchElement
from ._rats import RatsElement
from ._trail_choice import MODES, TrailChoiceInteraction


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
    branches = [BranchElement(label) for label in ('left', 'right')]
    evaporations = [
        ReactionInteraction(
            f'evaporation_{label}',
            reactants={(label, 'n'): 1},
            k=evaporation,
            mode=MODES[stochastic],
        )
        for label in ('left', 'right')
    ]
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
    return System(elements=branches, interactions=[*evaporations, *choices])


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
    alarm = AlarmInteraction(
        'alarm',
        calm='calm',
        nervous='nervous',
        spontaneous=spontaneous,
        contagion=contagion,
        stochastic=stochastic,
    )
    calming_reaction = ReactionInteraction(
        'calming',
        reactants={('nervous', 'n'): 1},
        products={('calm', 'n'): 1},
        k=calming,
        mode=MODES[stochastic],
    )
    return System(elements=groups, interactions=[alarm, calming_reaction])
