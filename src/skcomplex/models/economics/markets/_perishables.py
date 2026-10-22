"""Markets models: customers who choose between two shops of perishable goods by their past satisfaction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt
from scipy.special import expit

from ....core.base import Domain, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction
from ....core.laws import Law, Rule, Symbols
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView
from ....core.system import Observable, System

type Array = npt.NDArray[np.float64]

N_SHOPS = 2


def _asymmetry(shares: Array) -> float:
    """Return the asymmetry `m = p_1 - p_2` of the shares of customers of the first and the second shop."""
    return float(shares[0] - shares[1])


def _first_shop_probabilities(customers: list[AgentView], temperature: float) -> Array:
    """Return the probability `1 / (1 + exp(-(S_1 - S_2) / T))` that each customer visits the first shop."""
    difference = np.array([customer.s1 - customer.s2 for customer in customers], dtype=float)
    return expit(difference / temperature)


class Customer(Element):
    """Customer who remembers how satisfied it was with the products of each of two shops.

    Args:
        label:
            The label of the agent or group.

        s1:
            The satisfaction with the first shop.

        s2:
            The satisfaction with the second shop.

    Examples:
        >>> from skcomplex.models.economics.markets import Customer
        >>> Customer.group('customers', 10, s1=0.5, s2=0.5).size
        10
    """

    s1: float = state(default=0.5, domain=Domain.REAL)
    s2: float = state(default=0.5, domain=Domain.REAL)

    def laws(self, s: Symbols) -> list[Law]:
        """Return no laws, since the trade moves every customer.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


class Product(Element):
    """Product on the shelf of a shop, which ages from the time it was shelved.

    Args:
        label:
            The label of the agent or group.

        shop:
            The index of the shop, 0 or 1.

        shelved:
            The time at which the product was put on the shelf.

    Examples:
        >>> from skcomplex.models.economics.markets import Product
        >>> Product.group('products', 4, shop=[0, 0, 1, 1]).size
        4
    """

    shop: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)
    shelved: float = state(default=0.0, domain=Domain.REAL)

    def laws(self, s: Symbols) -> list[Law]:
        """Return no laws, since the trade sells and replaces products.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


class Shop(Element):
    """Shop of perishable goods, described by its share of customers and the mean freshness of its stock.

    Args:
        label:
            The label of the agent or group.

        share:
            The probability that a customer visits the shop, averaged over the customers.

        freshness:
            The mean freshness of the products on its shelves.

    Examples:
        >>> from skcomplex.models.economics.markets import Shop
        >>> Shop.group('shops', 2, share=0.5).size
        2
    """

    share: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    freshness: float = state(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return no laws, since the trade records the shares and the freshness.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


class PerishableTrade(Interaction):
    """Visits of customers to two shops, one random customer at a time, `N` visits per unit of time.

    A customer visits the first shop with the probability `1 / (1 + exp(-(S_1 - S_2) / T))` and buys a random product
    of age `tau`, whose freshness is `h = exp(-tau / tau_a)`. The shop replaces it with a fresh product, and the
    customer updates its satisfaction with the shop to `alpha S + (1 - alpha) h`. After the visits of a unit of time,
    each shop records its share of customers and the mean freshness of its stock.

    Args:
        label:
            The label of the interaction.

        customers:
            The path of the group of customers.

        products:
            The path of the group of products.

        shops:
            The path of the group of the two shops.

        T:
            The temperature of the choice, the randomness of the customers.

        alpha:
            The memory of the satisfaction, between 0 and 1.

        tau_a:
            The time over which the products age.

    Examples:
        >>> from skcomplex.models.economics.markets import PerishableTrade
        >>> PerishableTrade('trade', customers='customers', products='products', shops='shops', T=0.1).T
        0.1
    """

    customers: str = role()
    products: str = role()
    shops: str = role()
    T: float = parameter(default=0.1, domain=Domain.POSITIVE)
    alpha: float = parameter(default=0.9, domain=Domain.NONNEGATIVE)
    tau_a: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the trade rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.trade)]

    def trade(self, model: ModelView) -> None:
        """Run the visits of one unit of time and record the shares and the freshness of the shops.

        Args:
            model:
                The model.
        """
        temperature, alpha, tau_a = (model.parameter(name) for name in ('T', 'alpha', 'tau_a'))
        customers, products = model.agents(self.customers), model.agents(self.products)
        satisfaction = np.array([[customer.s1, customer.s2] for customer in customers], dtype=float)
        stock = np.array([product.shop for product in products], dtype=int)
        shelved = np.array([product.shelved for product in products], dtype=float)
        shelves = [np.flatnonzero(stock == shop) for shop in range(N_SHOPS)]
        rng, start, size = model.rng, model.time, len(customers)
        for visit in range(size):
            now = start + visit / size
            buyer = int(rng.integers(size))
            chosen = int(rng.random() >= expit((satisfaction[buyer, 0] - satisfaction[buyer, 1]) / temperature))
            sold = int(rng.choice(shelves[chosen]))
            freshness = np.exp(-(now - shelved[sold]) / tau_a)
            satisfaction[buyer, chosen] = alpha * satisfaction[buyer, chosen] + (1 - alpha) * freshness
            shelved[sold] = now
        for customer, (first, second) in zip(customers, satisfaction, strict=True):
            customer.s1, customer.s2 = float(first), float(second)
        for product, time in zip(products, shelved, strict=True):
            product.shelved = float(time)
        first = float(_first_shop_probabilities(customers, temperature).mean())
        freshness = np.exp(-(start + 1 - shelved) / tau_a)
        for record, share, shelf in zip(model.agents(self.shops), (first, 1 - first), shelves, strict=True):
            record.share = share
            record.freshness = float(freshness[shelf].mean())


@register_model(
    field='economics/markets',
    tags=('agents', 'discrete choice', 'symmetry breaking', 'phase transition', 'retail'),
    references=('Bertin 2021', 'Bouchaud 2013'),
    defaults={'until': 300},
    observables=lambda system: [Observable('m', group='shops', variable='share', reduce=_asymmetry)],
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
    discontinuous below. One unit of time is one visit per customer.

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
            The model, with the groups `customers`, `products`, and `shops`, and the asymmetry `m` of the shares of
            the two shops.

    Examples:
        >>> from skcomplex.models.economics.markets import perishable_goods_duopoly
        >>> perishable_goods_duopoly(20).elements[1].size
        20
    """
    rng = np.random.default_rng(random_state)
    shelf = max(round(stock * customers), 1)
    satisfaction = 0.5 + 0.01 * rng.standard_normal((N_SHOPS, customers))
    trade = PerishableTrade(
        'trade',
        customers='customers',
        products='products',
        shops='shops',
        T=T,
        alpha=alpha,
        tau_a=tau_a,
    )
    return System(
        components=[
            Customer.group('customers', customers, s1=satisfaction[0], s2=satisfaction[1]),
            Product.group('products', N_SHOPS * shelf, shop=np.repeat(np.arange(N_SHOPS), shelf).astype(float)),
            Shop.group('shops', N_SHOPS),
        ],
        interactions=[trade],
    )
