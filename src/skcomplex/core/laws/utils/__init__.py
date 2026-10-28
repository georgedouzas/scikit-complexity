"""Define what laws are written with besides Sage: reductions, draws, delays, rule tables, and calls."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....specs._draw_spec import bernoulli, categorical, draw, draws_in, normal, uniform
from ._agents import matches, random_pairing, transfer
from ._conditions import both, either, mod, where
from ._delays import delay_parts, distributed_delay
from ._history import delayed, derivative, lag
from ._lookups import lookup
from ._moments import moment
from ._operators import advection, divergence, gradient, inverse_laplacian, laplacian, partial, upwind
from ._quantum import dagger, has_operator, local_operator, operator_product
from ._reductions import (
    NEIGHBOR_REDUCTIONS,
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
from ._signals import pulse, pulse_train, ramp, series, step
from ._words import parse_word

__all__: list[str] = [
    'NEIGHBOR_REDUCTIONS',
    'advection',
    'bernoulli',
    'both',
    'categorical',
    'count',
    'dagger',
    'delay_parts',
    'delayed',
    'derivative',
    'distributed_delay',
    'divergence',
    'draw',
    'draws_in',
    'either',
    'gradient',
    'has_operator',
    'inverse_laplacian',
    'lag',
    'laplacian',
    'local_operator',
    'lookup',
    'matches',
    'maximum',
    'mean',
    'minimum',
    'mod',
    'moment',
    'neighborhoods_in',
    'neighbors',
    'normal',
    'operator_product',
    'parse_word',
    'partial',
    'pulse',
    'pulse_train',
    'ramp',
    'random_pairing',
    'reductions_in',
    'series',
    'shift_member',
    'step',
    'total',
    'transfer',
    'uniform',
    'upwind',
    'variance',
    'where',
]
