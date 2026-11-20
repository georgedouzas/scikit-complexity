"""Markets systems: customers who choose between two shops of perishable goods.

They choose by their past satisfaction.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.compile import PhaseSpec
from ....core.engine.solvers import AgentView, BaseSolver, MesaSolver, SystemView
from ....core.systems import BaseSystem, System
from ...general.agents import ChoiceInteraction, MovementInteraction
from ..common.learning import PayoffLearningInteraction

N_SHOPS = 2
PHASE = 'visit'


class CustomerView(AgentView):
    """Customer as its rules see it.

    It holds its satisfactions `s1` and `s2`, its `shop`, `freshness`, and temperature `T`.
    """

    s1: float
    s2: float
    shop: float
    freshness: float
    T: float


class CustomerElement(Element):
    """Customer who remembers how satisfied it was with each of two shops.

    The satisfaction is with the products of the shops. The customer visits the shop `k` of larger satisfaction `S_k`
    more often, with the logit utility `S_k / T`, and holds the freshness of the product it bought last.

    Args:
        label:
            The label of the agent or group.

        s1:
            The satisfaction with the first shop.

        s2:
            The satisfaction with the second shop.

        shop:
            The index of the shop the customer visits, 0 or 1.

        freshness:
            The freshness of the product the customer bought last.

        T:
            The temperature of the choice, the randomness of the customer.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.markets import CustomerElement
        >>> CustomerElement('customers', s1=0.5, s2=0.5, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        *,
        s1: object = 0.5,
        s2: object = 0.5,
        shop: Annotated[object, FiniteEnumeratedSet([0, 1])] = 0.0,
        freshness: Annotated[object, RealSet.closed(0, 1)] = 1.0,
        T: Annotated[object, RealSet.open(0, oo)] = 0.1,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.s1 = s1
        self.s2 = s2
        self.shop = shop
        self.freshness = freshness
        self.T = T

    @classmethod
    def _get_view(cls) -> type[CustomerView]:
        """Return the view of a customer.

        It holds its satisfactions, shop, freshness, and temperature.
        """
        return CustomerView

    @staticmethod
    def get_utility(agent: CustomerView, system: SystemView, option: float) -> float:
        """Return the utility `S_k / T` of visiting shop `k`.

        Args:
            agent:
                The customer.

            system:
                The system view of the choice.

            option:
                The index of the shop.

        Returns:
            utility:
                The satisfaction with the shop divided by the temperature.
        """
        return (agent.s1 if option == 0 else agent.s2) / agent.T

    @staticmethod
    def get_destination(agent: AgentViewProtocol, system: SystemViewProtocol) -> int | None:
        """Return the node of the shop the customer chose.

        Args:
            agent:
                The customer.

            system:
                The system view of the movement.

        Returns:
            node:
                The node of the shop, or `None` to stay for an agent that is not a customer.
        """
        return int(agent.shop) if isinstance(agent, CustomerView) else None

    @staticmethod
    def get_payoff(agent: CustomerView, system: SystemView, option: int) -> float:
        """Return the freshness of the product the customer bought last.

        It is the payoff of the shop it visited.

        Args:
            agent:
                The customer.

            system:
                The system view of the reinforcement.

            option:
                The index of the shop the customer visited.

        Returns:
            freshness:
                The freshness of the product.
        """
        return agent.freshness

    @staticmethod
    def get_preference(agent: CustomerView) -> float:
        """Return `p_1 - p_2 = tanh((S_1 - S_2) / 2T)`.

        It is how much more likely the customer visits the first shop.

        Args:
            agent:
                The customer.

        Returns:
            preference:
                The difference of the probabilities of visiting the first and the second shop.
        """
        return float(np.tanh((agent.s1 - agent.s2) / (2 * agent.T)))


class ProductView(AgentView):
    """Product as the purchase sees it.

    It holds the time it was `shelved` and the time `tau_a` over which it ages.
    """

    shelved: float
    tau_a: float


class ProductElement(Element):
    """Product on the shelf of a shop, which ages from the time it was shelved.

    Its freshness is `exp(-age / tau_a)`.

    Args:
        label:
            The label of the agent or group.

        shelved:
            The time at which the product was put on the shelf.

        tau_a:
            The time over which the product ages.

        values:
            Further values by name, such as the `node` of the shop of every member.

        size:
            The number of members, or `None` for one.

        space:
            The network whose nodes are the shops, or `None` for none.

    Examples:
        >>> from skcomplex.systems.economics.markets import ProductElement
        >>> ProductElement('products', values={'node': [0, 0, 1, 1]}, size=4).size
        4
    """

    def __init__(
        self,
        label: str,
        *,
        shelved: object = 0.0,
        tau_a: Annotated[object, RealSet.open(0, oo)] = 1.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.shelved = shelved
        self.tau_a = tau_a

    @classmethod
    def _get_view(cls) -> type[ProductView]:
        """Return the view of a product, with its shelving time and ageing time."""
        return ProductView


class PerishableGoodsDuopolySystem(BaseSystem):
    """Customers who choose between two shops of perishable goods.

    They prefer the one that sold fresher goods. Each shop holds `stock` products per customer, and every sale replaces
    a product with a fresh one, so the shop with more customers sells fresher goods and attracts more customers. With
    the control parameter `R = 2 stock / tau_a`, the symmetric state `m = p_1 - p_2 = 0` loses its stability below the
    temperature `T_c = R / (1 + R)^2`, and one shop takes most customers. The transition is continuous for
    `R > sqrt(3) - 1` and discontinuous below. One unit of time is `N` visits by customers drawn with replacement. A
    visited customer chooses a shop with the probability `1 / (1 + exp(-(S_1 - S_2) / T))` for the first one, moves to
    it, buys a random product of age `tau`, of freshness `h = exp(-tau / tau_a)`, which the shop replaces with a fresh
    one, and updates its satisfaction with that shop alone to `alpha S + (1 - alpha) h`. The shops are the two nodes of
    a network on which their products stand.

    The system has the groups `customers` and `products`, and the asymmetry `m` of the probabilities of visiting the two
    shops.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.markets import PerishableGoodsDuopolySystem
        >>> PerishableGoodsDuopolySystem(20).prepare().system_.elements[0].size
        20
    """

    field = 'economics/markets'
    topics = ('agents', 'discrete choice', 'symmetry breaking', 'phase transition', 'retail')
    references = ('Bertin 2021', 'Bouchaud 2013')
    until = 300
    observables = (Observable('m', group='customers', expression=CustomerElement.get_preference),)

    def __init__(
        self,
        customers: int = 200,
        *,
        stock: float = 0.5,
        T: float = 0.1,
        alpha: float = 0.9,
        tau_a: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.customers = customers
        self.stock = stock
        self.T = T
        self.alpha = alpha
        self.tau_a = tau_a
        self.random_state = random_state

    @staticmethod
    def _buy(customer: CustomerView, system: SystemView) -> None:
        """Buy a random product of the shop of a customer, which the shop replaces with a fresh one.

        The customer takes the freshness `exp(-(t - shelved) / tau_a)` of the product, and the shelving time of the
        product is reset to the time of the sale.
        """
        groups = set(system.get_groups('products'))
        products = [other for other in customer.get_agents_here() if other.group in groups]
        if len(products) > 1:
            products = [products[int(index)] for index in system.rng.permutation(len(products))[:1]]
        for product in products:
            if customer.alive and product.alive and isinstance(product, ProductView):
                restock = float(product.shelved - system.time)
                fresh = float(np.exp(-(system.time - product.shelved) / product.tau_a)) - customer.freshness
                product.shelved -= restock
                customer.freshness += fresh

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        shelf = max(round(self.stock * self.customers), 1)
        space = NetworkSpace(nx.empty_graph(N_SHOPS))
        satisfaction = 0.5 + 0.01 * rng.standard_normal((N_SHOPS, self.customers))
        choice = ChoiceInteraction(
            'choice',
            target='customers',
            state='shop',
            options=tuple(float(shop) for shop in range(N_SHOPS)),
            utility=CustomerElement.get_utility,
            schedule='member',
            phase=PHASE,
        )
        movement = MovementInteraction(
            'movement',
            target='customers',
            step=CustomerElement.get_destination,
            schedule='member',
            phase=PHASE,
        )
        purchase = Interaction(
            'purchase',
            roles={'target': 'customers', 'products': 'products'},
            laws=lambda s: [UpdateLaw(s.target, PerishableGoodsDuopolySystem._buy, phase=PHASE)],
        )
        satisfaction_update = PayoffLearningInteraction(
            'satisfaction',
            target='customers',
            scores=('s1', 's2'),
            payoff=CustomerElement.get_payoff,
            rate=1 - self.alpha,
            chosen='shop',
            schedule='member',
            phase=PHASE,
        )
        self.system_ = System(
            self.label,
            elements=[
                CustomerElement('customers', s1=satisfaction[0], s2=satisfaction[1], T=self.T, size=self.customers),
                ProductElement(
                    'products',
                    tau_a=self.tau_a,
                    values={'node': np.repeat(np.arange(N_SHOPS), shelf)},
                    size=N_SHOPS * shelf,
                    space=space,
                ),
            ],
            interactions=[choice, movement, purchase, satisfaction_update],
            space=space,
            solver=MesaSolver(phases=[PhaseSpec(PHASE, activation='visits', groups=('customers',))]),
        )
