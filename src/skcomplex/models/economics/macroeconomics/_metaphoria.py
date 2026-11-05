"""Metaphoria: a mortal population that lives as hunter-gatherers or in a monetary economy of firms and a bank."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np

from ....analysis import DistributionAnalysis
from ....core.elements import GroupElement
from ....core.interactions import BaseInteraction, TransferInteraction
from ....core.spaces import DiscreteTime, LatticeSpace
from ....core.system import System
from ....specs import (
    ObservableSpec,
    PartnerSpec,
    PhaseSpec,
    PhasesScheduleSpec,
)
from ....types import Array
from ..._catalog import register_model
from ..common.resources import RegrowthInteraction
from ._business import BusinessElement
from ._central_bank import CentralBankElement
from ._commons import CommonsInteraction
from ._estate import EstateInteraction
from ._food_market import FoodMarketInteraction
from ._labour_market import LabourMarketInteraction
from ._plot import PlotElement
from ._settler import ALIVE, PROPENSITIES, SettlerElement

MODES = ('hunter_gatherer', 'monetary')


PHASES = ('growth', 'work', 'banking', 'hiring', 'market', 'living', 'census')


SIGHT = 5.0


def _bank_balance(bank: SimpleNamespace) -> float:
    """Return the money the bank has issued plus the loans it holds, zero when every unit of money is a loan."""
    return float(bank.money + bank.loans)


def _living(fates: object) -> float:
    """Return the number of living people from their fates."""
    return float(np.sum(np.asarray(fates) == ALIVE))


def _money_gini(money: object) -> float:
    """Return the Gini coefficient of the money of people."""
    return DistributionAnalysis('gini').prepare(np.asarray(money, dtype=float)).run().gini_


def _metaphoria_observables(system: System) -> list[ObservableSpec]:
    """Return the living population, the Gini coefficient of money, and the bank balance of a monetary economy."""
    observables = [
        ObservableSpec('population', group='people', variable='fate', reduce=_living),
        ObservableSpec('money_gini', group='people', variable='money', reduce=_money_gini),
    ]
    if any(element.label == 'bank' for element in system.elements):
        observables.append(ObservableSpec('bank_balance', _bank_balance, group='bank', reduce='sum'))
    return observables


@register_model(
    field='economics/macroeconomics',
    tags=('agents', 'mortality', 'evolution', 'money', 'hunter-gatherers', 'stock-flow consistent'),
    references=('Gooding 2019',),
    defaults={'until': 300},
    observables=_metaphoria_observables,
)
def metaphoria(
    n_people: int = 100,
    *,
    mode: str = 'hunter_gatherer',
    shape: Sequence[int] = (20, 20),
    max_food: float = 15.0,
    max_resistance: float = 5.0,
    regrowth: float = 1.0,
    need: float = 2.0,
    labour: float = 5.0,
    interest: float = 0.003,
    repayment: float = 0.004,
    n_firms: int = 5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Gooding's Metaphoria, a mortal population of hunter-gatherers or of a monetary economy.

    Patches of a periodic lattice hold food that grows back to a random capacity and resist harvesting with a random
    resistance. People need food every step, age, die of starvation or old age, and have children who inherit their
    propensities to breed, explore, found firms, and quit jobs, with mutations, so the propensities evolve under the
    test of survival. Hunter-gatherers pool the labour of the adults on a patch and share the harvest. In the monetary
    economy firms harvest with paid workers, sell the food in shops, and borrow the wages from a bank that creates
    money as loans, so the money of all holders sums to zero at every step. Gooding found the hunter-gatherers stable,
    and the monetary population smaller and less healthy, with money concentrating to a Gini coefficient near 0.5. On
    land where the hunter-gatherers thrive the monetary population dwindles towards extinction, and with abundant food
    it lives on with persistent starvation.

    Args:
        n_people:
            The initial number of adults.

        mode:
            `'hunter_gatherer'` or `'monetary'`.

        shape:
            The number of patches along each axis.

        max_food:
            The largest capacity of a patch, drawn uniformly from 0.

        max_resistance:
            The largest resistance of a patch, drawn uniformly from 0.

        regrowth:
            The share of the capacity that grows back in a step.

        need:
            The food a person eats in a step.

        labour:
            The labour of an adult in a step.

        interest:
            The interest rate of loans per step.

        repayment:
            The share of the principal repaid per step.

        n_firms:
            The number of firms of the monetary economy at the start, each without an owner until it has a worker.

        random_state:
            The seed or generator of the patches, the places, ages, and propensities of the people.

    Returns:
        system:
            The model, with the group `patches`, one per cell, the variable group `people`, the reporters
            `census.births`, `census.starvations`, and `census.old_age`, and the observables `population` and
            `money_gini`. The monetary economy adds the variable group `businesses`, the group `bank`, and the reporter
            `shops.gdp`.

    Raises:
        ValueError:
            The mode is unknown.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import metaphoria
        >>> [group.label for group in metaphoria(10, mode='monetary', shape=(5, 5)).elements]
        ['patches', 'people', 'businesses', 'bank']
    """
    if mode not in MODES:
        error_msg = f'Unknown mode {mode!r}. Use one of {MODES}.'
        raise ValueError(error_msg)
    rng = np.random.default_rng(random_state)
    shape = tuple(int(item) for item in shape)
    space = LatticeSpace(shape)
    sites = list(np.ndindex(*shape))
    capacity = rng.uniform(0.0, max_food, len(sites))
    patches = GroupElement(
        'patches',
        element=PlotElement,
        size=len(sites),
        values={
            'food': capacity,
            'capacity': capacity,
            'resistance': rng.uniform(0.0, max_resistance, len(sites)),
            'regrowth': regrowth,
        },
        location=sites,
    )
    cells = rng.integers(0, shape, size=(n_people, len(shape)))
    traits: dict[str, Array] = {name: rng.uniform(0.0, bound, n_people) for name, bound in PROPENSITIES.items()}
    people = GroupElement(
        'people',
        element=SettlerElement,
        size=n_people,
        values={
            'age': rng.integers(15, 40, n_people).astype(float),
            'need': need,
            'labour': labour,
            'monetary': float(mode == 'monetary'),
            'sight': SIGHT,
            **traits,
        },
        variable=True,
        location=[tuple(int(item) for item in cell) for cell in cells],
    )
    firms = rng.integers(0, shape, size=(n_firms, len(shape)))
    businesses = GroupElement(
        'businesses',
        element=BusinessElement,
        size=n_firms,
        values={'interest': interest, 'repayment': repayment},
        variable=True,
        location=[tuple(int(item) for item in cell) for cell in firms],
    )
    bank = GroupElement('bank', element=CentralBankElement, size=1, location=(0,) * len(shape))
    economy = [businesses, bank] if mode == 'monetary' else []
    jobs = LabourMarketInteraction(
        'jobs',
        requesters='people',
        providers='businesses',
        rule=PartnerSpec(SIGHT, 'score', score=LabourMarketInteraction.score, clearing='provider'),
        limit=('hireable', 'limit'),
        quantity=('quantity', 'vacancies'),
        phase='hiring',
        on_match=LabourMarketInteraction.hire,
    )
    shops = FoodMarketInteraction(
        'shops',
        requesters='people',
        providers='businesses',
        rule=PartnerSpec(SIGHT, 'score', score=FoodMarketInteraction.score, clearing='provider'),
        limit=('limit', 'price'),
        quantity=('shortfall', 'stock'),
        allowance='money',
        transfers=(('money', -1),),
        phase='market',
        on_match=FoodMarketInteraction.sell,
    )
    dividends = TransferInteraction(
        'dividends',
        source='bank',
        target='people',
        state='money',
        amount=CentralBankElement.get_income,
        pairing='random',
        fallback='businesses',
        guard=lambda agent, holder, model: CentralBankElement.get_income(agent, holder, model) > 0,
        form='agent',
        phase='banking',
    )
    exchange: list[BaseInteraction] = (
        [CommonsInteraction('commons', people='people')] if mode == 'hunter_gatherer' else [jobs, dividends, shops]
    )
    return System(
        elements=[patches, people, *economy],
        interactions=[
            RegrowthInteraction(
                'regrowth',
                target='patches',
                state='food',
                growth='regrowth',
                relative=True,
                phase='growth',
            ),
            *exchange,
            EstateInteraction('census', people='people'),
        ],
        space=space,
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec(name) for name in PHASES])),
    )
