"""Define what laws are written with besides Sage.

They are reductions, pointwise functions, draws, delays, and calls.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._agents import matches, random_pairing, transfer
from ._cells import codifferential, exterior_derivative, hodge_star
from ._components import components
from ._conditions import both, either, mod, where
from ._contractions import contraction
from ._delays import distributed_delay
from ._draws import bernoulli, categorical, draw, normal, uniform
from ._history import delayed, derivative, lag, residence
from ._integrals import cumulative_integral, integral_transform, integration_variable
from ._local_operators import dagger, local_operator, operator_product
from ._moments import moment
from ._operators import (
    advection,
    collision_integral,
    convolve,
    displacement,
    divergence,
    gradient,
    inverse_helmholtz,
    laplacian,
    partial,
    shifted,
    upwind,
)
from ._paths import shortest_paths
from ._phase import coordinate, maximum_entropy, reduce_axes
from ._pointwise import clip, hill, logistic, sign, stop_gradient
from ._reductions import (
    circular_mean,
    count,
    hyperedges,
    log_sum_exp,
    maximum,
    mean,
    minimum,
    neighbors,
    reduction,
    shift_member,
    total,
    variance,
)
from ._signals import periodic_guard, pulse, pulse_train, ramp, series, step
from ._tensors import (
    antisymmetric_part,
    leray_projection,
    symmetric_part,
    tensor_divergence,
    vector_gradient,
)

__all__: list[str] = [
    'advection',
    'antisymmetric_part',
    'bernoulli',
    'both',
    'categorical',
    'circular_mean',
    'clip',
    'codifferential',
    'collision_integral',
    'components',
    'contraction',
    'convolve',
    'coordinate',
    'count',
    'cumulative_integral',
    'dagger',
    'delayed',
    'derivative',
    'displacement',
    'distributed_delay',
    'divergence',
    'draw',
    'either',
    'exterior_derivative',
    'gradient',
    'hill',
    'hodge_star',
    'hyperedges',
    'integral_transform',
    'integration_variable',
    'inverse_helmholtz',
    'lag',
    'laplacian',
    'leray_projection',
    'local_operator',
    'log_sum_exp',
    'logistic',
    'matches',
    'maximum',
    'maximum_entropy',
    'mean',
    'minimum',
    'mod',
    'moment',
    'neighbors',
    'normal',
    'operator_product',
    'partial',
    'periodic_guard',
    'pulse',
    'pulse_train',
    'ramp',
    'random_pairing',
    'reduce_axes',
    'reduction',
    'residence',
    'series',
    'shift_member',
    'shifted',
    'shortest_paths',
    'sign',
    'step',
    'stop_gradient',
    'symmetric_part',
    'tensor_divergence',
    'total',
    'transfer',
    'uniform',
    'upwind',
    'variance',
    'vector_gradient',
    'where',
]
