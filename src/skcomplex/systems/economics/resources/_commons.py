"""Natural resources systems: herders whose cows graze a shared meadow, the tragedy of the commons."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.interactions import LifeCycleInteraction, MapInteraction
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.system import System
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.specs import (
    ObservableSpec,
    PartnerSpec,
    PhaseSpec,
    PhasesScheduleSpec,
)
from ..._catalog import register_system
from ._cow import CowElement
from ._grazing import GrazingInteraction
from ._herder import UNLIMITED, HerderElement
from ._herder_view import HerderView
from ._pasture import PastureElement
from ._quota_regulator import QuotaRegulatorInteraction


def _commons_observables(system: System) -> list[ObservableSpec]:
    """Return the grass on the meadow, the size of the herd, and the mean quota of the herders."""
    del system
    return [
        ObservableSpec('grass', group='meadow', variable='grass', reduce='sum'),
        ObservableSpec('herd', group='cows', variable='energy', reduce='count'),
        ObservableSpec('quota', group='herders', variable='quota', reduce='mean'),
    ]


def _pays(herder: AgentViewProtocol, system: SystemViewProtocol) -> bool:
    """Return whether a cow of a herder earned more than its upkeep in the last step while the herd is below quota."""
    del system
    if not isinstance(herder, HerderView):
        return False
    earning = herder.income / herder.herd if herder.herd else herder.upkeep + 1
    return earning > herder.upkeep and herder.herd < herder.quota


def _exceeds(herder: AgentViewProtocol, system: SystemViewProtocol) -> bool:
    """Return whether the herd of a herder in the last step exceeds its quota, so it sells a cow."""
    del system
    return isinstance(herder, HerderView) and herder.herd > herder.quota


def _owns(herder: AgentViewProtocol, cow: AgentViewProtocol) -> bool:
    """Return whether a herder owns a cow, which the reference state `owner` of the cow names."""
    return cow.referenced('owner') is herder


def _get_calf(herder: AgentViewProtocol | None, system: SystemViewProtocol) -> dict[str, object]:
    """Return the owner and the energy of a new cow of a herder."""
    del system
    if not isinstance(herder, HerderView):
        return {}
    return {'owner': float(herder.id), 'energy': herder.newborn}


def _get_cell(herder: AgentViewProtocol | None, system: SystemViewProtocol) -> tuple[int, ...]:
    """Return a random cell of the meadow, where a new cow starts."""
    del herder
    return tuple(int(system.rng.integers(int(size))) for size in system.dimensions or ())


def _get_cleared(herder: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return zero, the income and the herd a herder starts a step with."""
    del herder, system
    return 0.0


def _get_investment() -> list[LifeCycleInteraction | MapInteraction]:
    """Return the member interactions of a herder: buy or sell a cow, then clear its income and herd."""
    return [
        LifeCycleInteraction(
            'investment',
            target='herders',
            group='cows',
            birth=_pays,
            values=_get_calf,
            placement=_get_cell,
            cull=_exceeds,
            kin=_owns,
            schedule='member',
            phase='herding',
        ),
        *(
            MapInteraction(
                f'{state}_clearing',
                target='herders',
                state=state,
                function=_get_cleared,
                form='agent',
                schedule='member',
                phase='herding',
            )
            for state in ('income', 'herd')
        ),
    ]


@register_system(
    field='economics/resources',
    tags=('commons', 'agents', 'overgrazing', 'regulation', 'grid'),
    references=('Hamill and Gilbert 2016 sec. 10.2', 'Hardin 1968'),
    defaults={'until': 100},
    observables=_commons_observables,
)
def tragedy_of_the_commons(
    n_herders: int = 20,
    *,
    width: int = 20,
    height: int = 20,
    cows: int = 1,
    quota: float | None = None,
    regrowth: float = 0.2,
    capacity: float = 2.0,
    upkeep: float = 0.3,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return herders whose cows graze a shared meadow, each adding cows while a cow pays, until the grass collapses.

    Every step every cow moves to the richest pasture it can reach, eats, and earns its herder milk, and the grass
    grows back logistically. Every herder adds a cow whenever a cow earned more than its upkeep, since the gain of a
    cow goes to its herder while the cost of the grass it eats falls on all. With more herders than the meadow
    carries, the herd grows past what the grass sustains, the grass collapses below a tenth of its capacity, and the
    cows starve, so the herd shrinks after it. A regulator with an adaptive quota of cows per herder, lowered when the
    grass runs short and raised when it recovers, keeps the meadow from collapse (Hamill and Gilbert 2016 sec. 10.2).

    Args:
        n_herders:
            The number of herders.

        width:
            The number of columns of the meadow.

        height:
            The number of rows of the meadow.

        cows:
            The initial number of cows of every herder.

        quota:
            The initial quota of cows per herder of a regulator that adapts it, or `None` for no regulation.

        regrowth:
            The logistic growth rate of the grass.

        capacity:
            The most grass a cell holds.

        upkeep:
            The cost of keeping a cow for a step.

        random_state:
            The seed or generator of the cells of the cows.

    Returns:
        system:
            The system, with the groups `meadow`, `cows`, and `herders`, whose adaptive parameter `quota` the regulator
            `regulator` changes, and the grazing `grazing`.

    Examples:
        >>> from skcomplex.systems.economics.resources import tragedy_of_the_commons
        >>> [group.size for group in tragedy_of_the_commons(4, width=5, height=5).elements]
        [25, 4, 4]
    """
    rng = np.random.default_rng(random_state)
    cells = [(x, y) for x in range(width) for y in range(height)]
    meadow = GroupElement(
        'meadow',
        element=PastureElement,
        size=len(cells),
        values={'grass': capacity, 'capacity': capacity, 'regrowth': regrowth},
        location=cells,
    )
    herders = GroupElement(
        'herders',
        element=HerderElement,
        size=n_herders,
        values={
            'upkeep': upkeep,
            'quota': UNLIMITED if quota is None else quota,
        },
        location=[cells[0]] * n_herders,
    )
    n_cows = n_herders * cows
    herd = GroupElement(
        'cows',
        element=CowElement,
        size=n_cows,
        values={'owner': np.repeat(np.arange(n_herders, dtype=float), cows)},
        variable=True,
        location=[cells[index] for index in rng.integers(len(cells), size=n_cows)],
    )
    regulator = QuotaRegulatorInteraction('regulator', phase='herding', regulated=float(quota is not None))
    grazing = GrazingInteraction(
        'grazing',
        requesters='cows',
        providers='meadow',
        rule=PartnerSpec('local', 'score', score=GrazingInteraction.score, clearing='provider'),
        herders='herders',
    )
    return System(
        elements=[meadow, herd, herders],
        interactions=[grazing, *_get_investment(), regulator],
        space=LatticeSpace((width, height), boundary='periodic', neighborhood='moore'),
        time=DiscreteTime(
            schedule=PhasesScheduleSpec([PhaseSpec('grazing'), PhaseSpec('growth'), PhaseSpec('herding')]),
        ),
    )
