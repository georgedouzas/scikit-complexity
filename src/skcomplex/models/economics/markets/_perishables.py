"""Markets models: customers who choose between two shops of perishable goods by their past satisfaction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import networkx as nx
import numpy as np

from ....core.elements import GroupElement
from ....core.interactions import ChoiceInteraction, MovementInteraction, ReinforcementInteraction, TransferInteraction
from ....core.laws import Sink, Source
from ....core.spaces import DiscreteTime, NetworkSpace
from ....core.system import System
from ....specs import ObservableSpec, PhaseSpec, PhasesScheduleSpec
from ..._catalog import register_model
from ._customer import CustomerElement
from ._product import ProductElement

N_SHOPS = 2


PHASE = 'visit'


@register_model(
    field='economics/markets',
    tags=('agents', 'discrete choice', 'symmetry breaking', 'phase transition', 'retail'),
    references=('Bertin 2021', 'Bouchaud 2013'),
    defaults={'until': 300},
    observables=lambda system: [ObservableSpec('m', group='customers', expression=CustomerElement.preference)],
)
def perishable_goods_duopoly(
    customers: int = 200,
    *,
    stock: float = 0.5,
    T: float = 0.1,
    alpha: float = 0.9,
    tau_a: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return customers who choose between two shops of perishable goods, preferring the one that sold fresher goods.

    Each shop holds `stock` products per customer, and every sale replaces a product with a fresh one, so the shop
    with more customers sells fresher goods and attracts more customers. With the control parameter
    `R = 2 stock / tau_a`, the symmetric state `m = p_1 - p_2 = 0` loses its stability below the temperature
    `T_c = R / (1 + R)^2`, and one shop takes most customers. The transition is continuous for `R > sqrt(3) - 1` and
    discontinuous below. One unit of time is `N` visits by customers drawn with replacement. A visited customer
    chooses a shop with the probability `1 / (1 + exp(-(S_1 - S_2) / T))` for the first one, moves to it, buys a random
    product of age `tau`, of freshness `h = exp(-tau / tau_a)`, which the shop replaces with a fresh one, and updates
    its satisfaction with that shop alone to `alpha S + (1 - alpha) h`. The shops are the two nodes of a network on
    which their products stand.

    Args:
        customers:
            The number of customers `N`.

        stock:
            The number of products of each shop per customer.

        T:
            The temperature of the choice, the randomness of the customers.

        alpha:
            The memory of the satisfaction, between 0 and 1.

        tau_a:
            The time over which the products age.

        random_state:
            The seed or generator of the initial satisfactions, which scatter by 0.01 around 0.5.

    Returns:
        system:
            The model, with the groups `customers` and `products`, and the asymmetry `m` of the probabilities of
            visiting the two shops.

    Examples:
        >>> from skcomplex.models.economics.markets import perishable_goods_duopoly
        >>> perishable_goods_duopoly(20).elements[0].check().prepare().size_
        20
    """
    rng = np.random.default_rng(random_state)
    shelf = max(round(stock * customers), 1)
    satisfaction = 0.5 + 0.01 * rng.standard_normal((N_SHOPS, customers))
    choice = ChoiceInteraction(
        'choice',
        target='customers',
        state='shop',
        options=tuple(float(shop) for shop in range(N_SHOPS)),
        utility=CustomerElement.utility,
        schedule='member',
        form='agent',
        phase=PHASE,
    )
    movement = MovementInteraction(
        'movement',
        target='customers',
        step=CustomerElement.destination,
        schedule='member',
        phase=PHASE,
    )
    purchase = TransferInteraction(
        'purchase',
        source='products',
        target='customers',
        state='shelved',
        into=Sink(),
        amount=ProductElement.restock,
        legs=[(Source(), 'freshness', ProductElement.sell)],
        pairing='cell',
        form='agent',
        schedule='member',
        actor='target',
        partners=1,
        phase=PHASE,
    )
    satisfaction_update = ReinforcementInteraction(
        'satisfaction',
        target='customers',
        scores=('s1', 's2'),
        payoff=CustomerElement.payoff,
        rate=1 - alpha,
        chosen='shop',
        schedule='member',
        form='agent',
        phase=PHASE,
    )
    return System(
        elements=[
            GroupElement(
                'customers',
                element=CustomerElement,
                size=customers,
                values={'s1': satisfaction[0], 's2': satisfaction[1], 'T': T},
            ),
            GroupElement(
                'products',
                element=ProductElement,
                size=N_SHOPS * shelf,
                values={'tau_a': tau_a},
                location=np.repeat(np.arange(N_SHOPS), shelf).tolist(),
            ),
        ],
        interactions=[choice, movement, purchase, satisfaction_update],
        space=NetworkSpace(nx.empty_graph(N_SHOPS)),
        time=DiscreteTime(
            schedule=PhasesScheduleSpec([PhaseSpec(PHASE, activation='visits', groups=('customers',))]),
        ),
    )
