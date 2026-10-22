"""Natural resources models: herders whose cows graze a shared meadow, the tragedy of the commons."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ....core.backends import AgentView, ModelView
from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction, Match, matches
from ....core.laws import Law, Rule, Symbols, transfer
from ....core.runtime import register_model
from ....core.space import DiscreteTime, LatticeSpace, Phase, Phases
from ....core.system import Observable, System
from ..realizations import Institution

UNLIMITED = 1e9


def _commons_observables(system: System) -> list[Observable]:
    """Return the grass on the meadow, the size of the herd, and the mean quota of the herders."""
    del system
    return [
        Observable('grass', group='meadow', variable='grass', reduce='sum'),
        Observable('herd', group='cows', variable='energy', reduce='count'),
        Observable('quota', group='herders', variable='quota', reduce='mean'),
    ]


@dataclass(frozen=True)
class RichestPasture:
    """Pairing of every cow, in random order, with the free pasture of most grass in its cell or the cells around it.

    Examples:
        >>> from skcomplex.models.economics.resources import RichestPasture
        >>> RichestPasture()
        RichestPasture()
    """

    def __call__(
        self,
        left: Sequence[Any],
        right: Sequence[Any],
        rng: np.random.Generator,
        capacity: int,
    ) -> list[tuple[Any, Any]]:
        """Return the pairs of cows and pastures.

        Args:
            left:
                The cows.

            right:
                The pastures, one per cell.

            rng:
                The random generator.

            capacity:
                The largest number of cows on a pasture.

        Returns:
            pairs:
                The pairs of cows and the pastures they graze.
        """
        pastures = {tuple(int(item) for item in np.atleast_1d(pasture.location)): pasture for pasture in right}
        taken: dict[int, int] = {}
        pairs: list[tuple[Any, Any]] = []
        for index in rng.permutation(len(left)):
            cow = left[int(index)]
            cells = [cow.location, *cow.neighbor_cells()]
            options = [pastures.get(tuple(int(item) for item in np.atleast_1d(cell))) for cell in cells]
            free = [pasture for pasture in options if pasture is not None and taken.get(pasture.id, 0) < capacity]
            if not free:
                continue
            best = max(free, key=lambda pasture: pasture.grass)
            taken[best.id] = taken.get(best.id, 0) + 1
            pairs.append((cow, best))
        return pairs


class Pasture(Element):
    """Cell of the meadow whose grass grows back logistically, from a seed bank when it is grazed bare.

    Args:
        label:
            The label of the agent or group.

        grass:
            The grass on the cell.

        capacity:
            The most grass the cell holds.

        regrowth:
            The logistic growth rate of the grass.

        seed:
            The share of the capacity that grows back as if it stood on a bare cell.

    Examples:
        >>> from skcomplex.models.economics.resources import Pasture
        >>> Pasture.group('meadow', 4, capacity=2.0).size
        4
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'grass', 'color_range': (0.0, 2.0)})
    grass: float = state(default=2.0, domain=Domain.NONNEGATIVE)
    capacity: float = state(default=2.0, domain=Domain.POSITIVE)
    regrowth: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    seed: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the regrowth rule of the phase of growth.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.grow, phase='growth')]

    def grow(self, agent: AgentView, model: ModelView) -> None:
        """Grow the grass logistically up to the capacity.

        Args:
            agent:
                The pasture.

            model:
                The model.
        """
        stand = max(agent.grass, model.parameter('seed') * agent.capacity)
        growth = model.parameter('regrowth') * stand * (1 - stand / agent.capacity)
        agent.grass = min(agent.capacity, agent.grass + growth)


class Cow(Element):
    """Cow of a herder that gains the energy of the grass it eats, burns its metabolism, and starves without food.

    Args:
        label:
            The label of the agent or group.

        owner:
            The number of the herder that owns the cow.

        energy:
            The energy of the cow.

    Examples:
        >>> from skcomplex.models.economics.resources import Cow
        >>> Cow.group('cows', 3, energy=4.0).size
        3
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'energy'})
    owner: float = state(default=-1.0, domain=Domain.REAL)
    energy: float = state(default=4.0, domain=Domain.REAL)

    def laws(self, s: Symbols) -> list[Law]:
        """Return no laws, since the grazing moves the cows and their energy.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


class Herder(Element):
    """Herder who adds a cow whenever a cow earned more than its upkeep, and sells one when the herd exceeds its quota.

    Args:
        label:
            The label of the agent or group.

        number:
            The number of the herder, which its cows record as their owner.

        income:
            The milk income of the herder in the last step.

        herd:
            The number of cows of the herder in the last step.

        upkeep:
            The cost of keeping a cow for a step.

        newborn:
            The energy of a new cow.

        quota:
            The largest herd of a herder, an adaptive parameter that a regulator may change.

    Examples:
        >>> from skcomplex.models.economics.resources import Herder
        >>> Herder.group('herders', 5).size
        5
    """

    number: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    income: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    herd: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    upkeep: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    newborn: float = parameter(default=4.0, domain=Domain.POSITIVE)
    quota: float = parameter(default=UNLIMITED, domain=Domain.POSITIVE, adaptive=True)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rule that grows the herd in the phase of herding.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.invest, phase='herding')]

    def invest(self, agent: AgentView, model: ModelView) -> None:
        """Add a cow while a cow earns more than its upkeep below the quota, and sell a cow above the quota.

        Args:
            agent:
                The herder.

            model:
                The model.
        """
        earning = agent.income / agent.herd if agent.herd else model.parameter('upkeep') + 1
        if earning > model.parameter('upkeep') and agent.herd < agent.quota:
            model.create('cows', owner=agent.number, energy=model.parameter('newborn'))
        elif agent.herd > agent.quota:
            next(cow for cow in model.agents('cows') if cow.owner == agent.number).remove()
        agent.income = 0.0
        agent.herd = 0.0


class Grazing(Interaction):
    """Grazing of the meadow, where every cow moves to the richest pasture it can reach and eats its fill.

    The matching law pairs every cow with the free pasture of most grass in its cell or the cells around it, the cow
    moves there and eats up to its appetite as a transfer of grass to its energy, burns its metabolism, and earns its
    herder milk for what it ate. A cow whose energy runs out dies.

    Args:
        label:
            The label of the grazing.

        cows:
            The path of the group of cows.

        meadow:
            The path of the group of pastures.

        herders:
            The path of the group of herders.

        appetite:
            The most grass a cow eats in a step.

        metabolism:
            The energy a cow burns in a step.

        milk:
            The income of a herder per unit of grass its cows eat.

    Examples:
        >>> from skcomplex.models.economics.resources import Grazing
        >>> Grazing('grazing', cows='cows', meadow='meadow', herders='herders').appetite
        1.0
    """

    cows: str = role()
    meadow: str = role()
    herders: str = role()
    appetite: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    metabolism: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    milk: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Any]:
        """Return the matching of cows with pastures and the grazing rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            Match('cows', 'meadow', pairing=RichestPasture(), label='grazing', phase='grazing'),
            Rule(self.graze, phase='grazing'),
        ]

    def graze(self, model: Any) -> None:  # noqa: ANN401
        """Move every cow to its pasture, feed it, pay its herder, and let the starving cows die.

        Args:
            model:
                The model view of the grazing.
        """
        herders = {int(herder.number): herder for herder in model.agents('herders')}
        for cow, pasture in matches(model, 'grazing'):
            cow.move_to(pasture.location)
            eaten = min(pasture.grass, model.parameter('appetite'))
            transfer(pasture, cow, eaten, 'grass', 'energy')
            herders[int(cow.owner)].income += model.parameter('milk') * eaten
        for cow in model.agents('cows'):
            cow.energy -= model.parameter('metabolism')
            herders[int(cow.owner)].herd += 1
            if cow.energy < 0:
                cow.remove()


class Regulator(Institution):
    """Regulator that halves the quota of cows per herder when the grass runs short and raises it when it recovers.

    The quota is the adaptive parameter `quota` of every herder. It halves, down to one cow, when the grass falls below
    the share `low` of the capacity of the meadow, and grows by one cow when the grass is above the share `high`.
    Without regulation it stays as it is.

    Args:
        label:
            The label of the regulator.

        rule:
            Unused, since the regulator overrides `act`.

        phase:
            The phase the regulator acts in.

        regulated:
            One when the regulator changes the quota, zero otherwise.

        low:
            The share of the capacity of the meadow below which the quota halves.

        high:
            The share of the capacity of the meadow above which the quota rises by one cow.

    Examples:
        >>> from skcomplex.models.economics.resources import Regulator
        >>> Regulator('regulator', regulated=1.0).regulated
        1.0
    """

    regulated: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    low: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    high: float = parameter(default=0.8, domain=Domain.NONNEGATIVE)

    def act(self, model: Any) -> None:  # noqa: ANN401
        """Change the quota with the share of grass left on the meadow.

        Args:
            model:
                The model view of the regulator.
        """
        if not model.parameter('regulated'):
            return
        pastures = model.agents('meadow')
        share = sum(pasture.grass for pasture in pastures) / sum(pasture.capacity for pasture in pastures)
        for herder in model.agents('herders'):
            if share < model.parameter('low'):
                herder.quota = max(1.0, float(np.floor(herder.quota / 2)))
            elif share > model.parameter('high'):
                herder.quota += 1


@register_model(
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
            The model, with the groups `meadow`, `cows`, and `herders`, whose adaptive parameter `quota` the regulator
            `regulator` changes, and the grazing `grazing`.

    Examples:
        >>> from skcomplex.models.economics.resources import tragedy_of_the_commons
        >>> [group.size for group in tragedy_of_the_commons(4, width=5, height=5).elements]
        [25, 4, 4]
    """
    rng = np.random.default_rng(random_state)
    cells = [(x, y) for x in range(width) for y in range(height)]
    meadow = Pasture.group('meadow', len(cells), location=cells, grass=capacity, capacity=capacity, regrowth=regrowth)
    herders = Herder.group(
        'herders',
        n_herders,
        location=[cells[0]] * n_herders,
        number=np.arange(n_herders, dtype=float),
        upkeep=upkeep,
        quota=UNLIMITED if quota is None else quota,
    )
    n_cows = n_herders * cows
    herd = Cow.group(
        'cows',
        n_cows,
        variable=True,
        location=[cells[index] for index in rng.integers(len(cells), size=n_cows)],
        owner=np.repeat(np.arange(n_herders, dtype=float), cows),
    )
    regulator = Regulator('regulator', phase='herding', regulated=float(quota is not None))
    grazing = Grazing('grazing', cows='cows', meadow='meadow', herders='herders')
    return System(
        components=[meadow, herd, herders],
        interactions=[grazing, regulator],
        space=LatticeSpace((width, height), boundary='periodic', neighborhood='moore'),
        time=DiscreteTime(schedule=Phases([Phase('grazing'), Phase('growth'), Phase('herding')])),
    )
