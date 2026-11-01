"""Define what laws are written with besides Sage: reductions, pointwise functions, draws, delays, and calls."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....specs._draw_spec import bernoulli, categorical, draw, draws_in, normal, uniform
from ._agents import matches, random_pairing, transfer
from ._components import components
from ._conditions import both, either, mod, where
from ._contractions import contraction, contractions_in
from ._delays import delay1, delay3, delay_parts, distributed_delay, smooth
from ._history import delayed, derivative, lag, residence
from ._lookups import lookup
from ._moments import moment
from ._operators import (
    advection,
    convolve,
    displacement,
    divergence,
    gradient,
    inverse_helmholtz,
    inverse_laplacian,
    laplacian,
    partial,
    shifted,
    upwind,
)
from ._pointwise import clip, hill, logistic, sign
from ._quantum import dagger, has_operator, local_operator, operator_product
from ._reductions import (
    NEIGHBOR_REDUCTIONS,
    circular_mean,
    count,
    hyperedges,
    hyperedges_in,
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

__all__: list[str] = [
    'NEIGHBOR_REDUCTIONS',
    'advection',
    'bernoulli',
    'both',
    'categorical',
    'circular_mean',
    'clip',
    'components',
    'contraction',
    'contractions_in',
    'convolve',
    'count',
    'dagger',
    'delay1',
    'delay3',
    'delay_parts',
    'delayed',
    'derivative',
    'displacement',
    'distributed_delay',
    'divergence',
    'draw',
    'draws_in',
    'either',
    'gradient',
    'has_operator',
    'hill',
    'hyperedges',
    'hyperedges_in',
    'inverse_helmholtz',
    'inverse_laplacian',
    'lag',
    'laplacian',
    'local_operator',
    'logistic',
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
    'partial',
    'pulse',
    'pulse_train',
    'ramp',
    'random_pairing',
    'reductions_in',
    'residence',
    'series',
    'shift_member',
    'shifted',
    'sign',
    'smooth',
    'step',
    'total',
    'transfer',
    'uniform',
    'upwind',
    'variance',
    'where',
]
