"""Define what laws are written with besides Sage: reductions, draws, delays, rule tables, and calls."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._agents import matches, random_pairing, transfer
from ._call import Call
from ._delays import delay_parts, distributed_delay
from ._draws import Draw, bernoulli, categorical, draw, draws_in, normal, uniform
from ._moments import moment
from ._operators import advection, divergence, gradient, inverse_laplacian, laplacian, partial, upwind
from ._quantum import dagger, has_operator, local_operator, operator_product
from ._reductions import (
    NEIGHBOR_REDUCTIONS,
    NeighborReduction,
    Reduction,
    count,
    maximum,
    mean,
    minimum,
    neighborhoods_in,
    neighbors,
    reductions_in,
    shift_member,
    total,
    variance,
)
from ._tables import OuterTotalistic, Table, Totalistic
from ._words import parse_word

__all__: list[str] = [
    'NEIGHBOR_REDUCTIONS',
    'Call',
    'Draw',
    'NeighborReduction',
    'OuterTotalistic',
    'Reduction',
    'Table',
    'Totalistic',
    'advection',
    'bernoulli',
    'categorical',
    'count',
    'dagger',
    'delay_parts',
    'distributed_delay',
    'divergence',
    'draw',
    'draws_in',
    'gradient',
    'has_operator',
    'inverse_laplacian',
    'laplacian',
    'local_operator',
    'matches',
    'maximum',
    'mean',
    'minimum',
    'moment',
    'neighborhoods_in',
    'neighbors',
    'normal',
    'operator_product',
    'parse_word',
    'partial',
    'random_pairing',
    'reductions_in',
    'shift_member',
    'total',
    'transfer',
    'uniform',
    'upwind',
    'variance',
]
