"""Define what laws are written with besides Sage.

They are reductions, pointwise functions, draws, delays, and calls.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....specs._draw_spec import bernoulli, categorical, draw, draws_in, normal, uniform
from ._agents import matches, random_pairing, transfer
from ._cells import cell_operations_in, codifferential, exterior_derivative, hodge_star
from ._components import components
from ._conditions import both, either, mod, where
from ._contractions import contraction, contractions_in
from ._delays import delay_parts, distributed_delay
from ._history import delayed, derivative, lag, residence
from ._integrals import cumulative_integral, integral_transform, integration_variable
from ._local_operators import dagger, has_operator, local_operator, operator_product
from ._moments import moment
from ._operators import (
    INVERSION_BOUNDARIES,
    advection,
    collision_integral,
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
from ._paths import shortest_paths
from ._phase import COORDINATE_PREFIX, coordinate, coordinates_of, maximum_entropy, reduce_axes
from ._pointwise import clip, hill, logistic, sign, stop_gradient
from ._reductions import (
    MAX_NEIGHBOR_HYPEREDGE_SIZE,
    NEIGHBOR_REDUCTIONS,
    circular_mean,
    count,
    hyperedges,
    hyperedges_in,
    log_sum_exp,
    maximum,
    mean,
    minimum,
    neighborhoods_in,
    neighbors,
    reduction,
    reductions_in,
    shift_member,
    total,
    variance,
)
from ._signals import periodic_guard, pulse, pulse_train, ramp, series, step
from ._tensors import (
    antisymmetric_part,
    distribute_operators,
    leray_projection,
    symmetric_part,
    tensor_divergence,
    vector_gradient,
)

__all__: list[str] = [
    'COORDINATE_PREFIX',
    'INVERSION_BOUNDARIES',
    'MAX_NEIGHBOR_HYPEREDGE_SIZE',
    'NEIGHBOR_REDUCTIONS',
    'advection',
    'antisymmetric_part',
    'bernoulli',
    'both',
    'categorical',
    'cell_operations_in',
    'circular_mean',
    'clip',
    'codifferential',
    'collision_integral',
    'components',
    'contraction',
    'contractions_in',
    'convolve',
    'coordinate',
    'coordinates_of',
    'count',
    'cumulative_integral',
    'dagger',
    'delay_parts',
    'delayed',
    'derivative',
    'displacement',
    'distribute_operators',
    'distributed_delay',
    'divergence',
    'draw',
    'draws_in',
    'either',
    'exterior_derivative',
    'gradient',
    'has_operator',
    'hill',
    'hodge_star',
    'hyperedges',
    'hyperedges_in',
    'integral_transform',
    'integration_variable',
    'inverse_helmholtz',
    'inverse_laplacian',
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
    'neighborhoods_in',
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
    'reductions_in',
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
