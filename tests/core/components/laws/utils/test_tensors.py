"""Test the `_tensors` module of `skcomplex.core.components.laws.utils`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR, identity_matrix, var, vector

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import (
    antisymmetric_part,
    distribute_operators,
    inverse_laplacian,
    laplacian,
    leray_projection,
    partial,
    symmetric_part,
    tensor_divergence,
    vector_gradient,
)
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumpyBackend
from skcomplex.core.engine.compile import compile_system
from skcomplex.core.errors import TensorShapeError
from skcomplex.core.specs import StateSpec


def _vector_element(rate):
    """Return an element with a scalar `f`, a vector `g`, and a vector `w`, whose rate a function gives."""

    class VectorElement(Element):
        def __init__(
            self,
            label,
            w: Annotated[object, StateSpec(vector=True)] = 0.0,
            g: Annotated[object, StateSpec(vector=True)] = 0.0,
            f: Annotated[object, StateSpec()] = 0.0,
        ):
            super().__init__(label)
            self.w = w
            self.g = g
            self.f = f

        def _get_laws(self, namespace):
            return [RateLaw(namespace.w, rate(namespace))]

    return VectorElement


def _rate_of_w(rate, space, values, **settings):
    """Return the rate of the state `w` at its initial values, with one row per site."""
    field = FieldElement('m', element=_vector_element(rate), space=space, values=values, **settings)
    problem = compile_system(System(elements=[field]), backend=NumpyBackend())
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    return np.reshape(problem.states.read(drift, 'm.w'), (np.prod(space.shape), -1))


def test_tensor_divergence_of_a_diagonal_tensor_on_a_spectral_field():
    """The divergence of `sin(x) I` is `(cos(x), 0)` within 1e-10 at every site."""
    size = 64
    spacing = 2 * np.pi / size
    x = np.arange(size) * spacing
    space = LatticeSpace((size, size), spacing=spacing)
    field = np.repeat(np.sin(x)[:, None], size, axis=1)
    rate = _rate_of_w(
        lambda s: tensor_divergence(s.f * identity_matrix(SR, 2)),
        space,
        {'f': field},
        discretization='spectral',
    )
    expected = np.repeat(np.cos(x)[:, None], size, axis=1).ravel()
    assert np.max(np.abs(rate[:, 0] - expected)) < 1e-10
    assert np.max(np.abs(rate[:, 1])) < 1e-10


def test_symmetric_and_antisymmetric_parts_split_the_velocity_gradient():
    """The strain rate is symmetric, the vorticity tensor antisymmetric, and their sum the velocity gradient."""
    gradient = vector_gradient([var('u'), var('v'), var('w')])
    strain, vorticity = symmetric_part(gradient), antisymmetric_part(gradient)
    assert strain == strain.transpose()
    assert vorticity == -vorticity.transpose()
    assert (strain + vorticity - gradient).simplify_full().is_zero()


@pytest.mark.parametrize(
    ('utility', 'value'),
    [
        (tensor_divergence, var('u')),
        (symmetric_part, vector([var('u'), var('v')])),
        (antisymmetric_part, var('u')),
        (vector_gradient, var('u')),
        (leray_projection, var('u')),
    ],
)
def test_tensor_utilities_reject_an_expression_of_the_wrong_shape(utility, value):
    """A tensor utility given the wrong shape raises an error that names the utility."""
    with pytest.raises(TensorShapeError, match=utility.__name__):
        utility(value)


def test_distribute_operators_moves_numeric_factors_out():
    """A linear operator of a linear combination becomes a combination of operators."""
    a, b = var('a, b')
    assert distribute_operators(laplacian(-a - 2 * b) + laplacian(a)).expand() == -2 * laplacian(b)
    assert distribute_operators(partial(-a, 0)) == -partial(a, 0)


def test_leray_projection_is_divergence_free_in_three_dimensions():
    """The projection of a random smooth vector field has a discrete divergence below 1e-10 of its largest component."""
    size = 16
    spacing = 2 * np.pi / size
    generator = np.random.default_rng(0)
    axes = np.meshgrid(*[np.arange(size) * spacing] * 3, indexing='ij')
    modes = generator.integers(1, 4, size=(8, 3))
    phases = generator.uniform(0, 2 * np.pi, size=(8, 3))
    values = np.stack(
        [
            sum(
                np.sin(sum(mode[axis] * axes[axis] for axis in range(3)) + phase[component])
                for mode, phase in zip(modes, phases, strict=True)
            )
            for component in range(3)
        ],
        axis=-1,
    )
    space = LatticeSpace((size, size, size), spacing=spacing)
    rate = _rate_of_w(
        lambda s: leray_projection(s.g),
        space,
        {'g': values.reshape(-1, 3)},
        discretization='spectral',
    )
    spectra = [np.fft.fftn(rate[:, axis].reshape((size, size, size))) for axis in range(3)]
    wave = np.fft.fftfreq(size, spacing) * 2 * np.pi
    numbers = np.meshgrid(wave, wave, wave, indexing='ij')
    divergence = np.real(np.fft.ifftn(sum(1j * numbers[axis] * spectra[axis] for axis in range(3))))
    assert np.max(np.abs(divergence)) < 1e-10 * np.max(np.abs(rate))


def test_leray_projection_gives_the_poiseuille_profile_between_symmetric_walls():
    """The Stokes velocity of a uniform force odd about both walls matches the parabola within 1% of its peak."""
    width, cross = 64, 8
    spacing = 1.0 / (width / 2)
    y = np.arange(width) * spacing
    force = np.where(y < 1.0, 1.0, -1.0)
    force[[0, width // 2]] = 0.0
    forces = np.zeros((cross * width, 2))
    forces[:, 1] = 0.0
    forces[:, 0] = np.tile(force, cross)
    space = LatticeSpace((cross, width), spacing=spacing)
    rate = _rate_of_w(
        lambda s: leray_projection(s.g.apply_map(lambda item: inverse_laplacian(-item))),
        space,
        {'g': forces},
        discretization='spectral',
        symmetry={'g': (None, 'odd'), 'w': (None, 'odd')},
    )
    profile = rate[: width // 2 + 1, 0]
    parabola = y[: width // 2 + 1] * (1.0 - y[: width // 2 + 1]) / 2
    assert np.max(np.abs(profile - parabola)) < 0.01 * parabola.max()
    assert np.max(np.abs(rate[:, 1])) < 1e-10
