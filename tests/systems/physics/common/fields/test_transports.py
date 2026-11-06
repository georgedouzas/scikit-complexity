"""Test the transports of tensors and vectors in a flow."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR, identity_matrix, matrix, var, vector
from scipy.integrate import solve_ivp

from skcomplex.core.components.laws.utils import partial, symmetric_part, upwind, vector_gradient
from skcomplex.systems.physics.common.fields import corotational, upper_convected


def test_upper_convected_of_the_identity_is_minus_twice_the_strain_rate():
    """The upper-convected derivative of the constant identity is minus twice the symmetric velocity gradient."""
    velocity = [var('u'), var('v')]
    transport = upper_convected(identity_matrix(SR, 2), velocity)
    assert (transport + 2 * symmetric_part(vector_gradient(velocity))).simplify_full().is_zero()


def test_corotational_turns_a_tensor_with_half_the_vorticity():
    """A zero co-rotational derivative in a uniform vorticity turns the principal axis at half the vorticity."""
    omega = 0.8
    a, b, u, v = var('a, b, u, v')
    transport = corotational(matrix(SR, [[a, b], [b, -a]]), [u, v])
    flow = {partial(u, 0): 0, partial(u, 1): -omega / 2, partial(v, 0): omega / 2, partial(v, 1): 0}
    uniform = {upwind(item, axis, speed): 0 for item in (a, b) for axis, speed in enumerate((u, v))}
    rates = [(-transport[0, column]).subs(flow | uniform) for column in (0, 1)]

    def function(_, y):
        return [float(rate.subs(a=y[0], b=y[1])) for rate in rates]

    solution = solve_ivp(function, (0.0, 2.0), [0.5, 0.0], rtol=1e-11, atol=1e-12)
    angle = 0.5 * np.arctan2(solution.y[1, -1], solution.y[0, -1])
    assert angle == pytest.approx(omega / 2 * 2.0, rel=1e-6)


def test_corotational_turns_a_vector_with_half_the_vorticity():
    """A vector turns with the flow at half the vorticity, as a tensor does."""
    omega = 0.8
    p, q, u, v = var('p, q, u, v')
    transport = corotational(vector(SR, [p, q]), [u, v])
    flow = {partial(u, 0): 0, partial(u, 1): -omega / 2, partial(v, 0): omega / 2, partial(v, 1): 0}
    uniform = {upwind(item, axis, speed): 0 for item in (p, q) for axis, speed in enumerate((u, v))}
    rates = [(-item).subs(flow | uniform) for item in transport]
    assert bool(rates[0] == -omega / 2 * q)
    assert bool(rates[1] == omega / 2 * p)
