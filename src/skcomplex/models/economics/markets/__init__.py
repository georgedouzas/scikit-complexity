"""Markets: prices, competition, retail, inventories, and queues."""

from ._cobweb import Cobweb, Market, cobweb
from ._duopoly import CournotAdjustment, Duopolist, cournot_duopoly
from ._inventory import CarDealership, car_dealer
from ._prices import QuadraticSupplyMarket, riccati_prices
from ._produce import Shopper, Stall, produce_market
from ._queues import Car, CarWashOperation, Till, Washer, car_wash

__all__: list[str] = [
    'Car',
    'CarDealership',
    'CarWashOperation',
    'Cobweb',
    'CournotAdjustment',
    'Duopolist',
    'Market',
    'QuadraticSupplyMarket',
    'Shopper',
    'Stall',
    'Till',
    'Washer',
    'car_dealer',
    'car_wash',
    'cobweb',
    'cournot_duopoly',
    'produce_market',
    'riccati_prices',
]
