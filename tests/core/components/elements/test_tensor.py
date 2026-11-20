"""Test the `_tensor` module of `skcomplex.core.components.elements`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, Tensor
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import laplacian
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class OrderElement(Element):
    """Sites with a symmetric traceless order tensor."""

    def __init__(
        self,
        label: str,
        *,
        Q: Annotated[object, Tensor('traceless')] = 0.0,
        space: LatticeSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.Q = Q


class RelaxationInteraction(Interaction):
    """Relaxation `Q' = -Q (tr(Q^2) - S^2)` of an order tensor towards the order `S`."""

    def __init__(self, label: str, *, target: Role, S: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.S = S

    def _prepare(self, context: object = None) -> None:
        """Set the rate of the tensor, which reads it as a matrix."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            Q = context.target.Q
            square = (Q * Q).trace()
            self.laws_ = [RateLaw(Q, -(square - context.S**2 / 2) * Q)]


def test_tensor_holds_its_symmetry():
    """Test that a tensor holds the symmetry of its matrix."""
    assert Tensor('traceless').symmetry == 'traceless'


def test_tensor_rejects_an_unknown_symmetry():
    """Test that the check names the expected form of a tensor."""
    with pytest.raises(ValidationError, match=r"Tensor\('traceless'\)"):
        Tensor('diagonal')._check()


def test_tensor_state_holds_the_independent_components_and_stays_traceless():
    """Test that a traceless tensor of a field is read as a matrix and relaxes to its order with two components."""
    start = np.tile([0.2, 0.1], (4, 1))
    sites = OrderElement('sites', Q=start, space=LatticeSpace((2, 2)))
    system = System(elements=[sites], interactions=[RelaxationInteraction('relax', target='sites')]).prepare()
    frames = np.asarray(system.simulate(until=20.0, dt=10.0).get_values('sites.Q'))
    assert frames.shape == (3, 4, 2)
    order = 2 * (frames[-1] ** 2).sum(axis=1)
    assert np.allclose(order, 0.5, atol=1e-4)
    assert np.allclose(frames[-1, :, 1] / frames[-1, :, 0], 0.5)


def _decay(s):
    """Return the decay of the tensor `Q` at unit rate."""
    Q = s.target.Q
    return [RateLaw(Q, -Q)]


def _smooth(s):
    """Return the relaxation of the tensor `Q` by the Laplacian of every entry of its matrix."""
    Q = s.target.Q
    return [RateLaw(Q, Q.apply_map(laplacian))]


def _film(space, laws, **settings):
    """Return a system of an order tensor on a lattice under the laws of one interaction."""
    rate = Interaction('rate', roles={'target': 'film'}, laws=laws)
    film = OrderElement('film', Q=settings.pop('Q', 0.0), space=space)
    return System(elements=[film], interactions=[rate], **settings)


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_rate_law_relaxes_a_tensor_state(backend):
    """Test that a decay of the matrix of a tensor gives every independent component its rate."""
    start = np.tile([1.0, 0.5], (9, 1))
    system = _film(LatticeSpace((3, 3)), _decay, Q=start, backend=backend()).prepare()
    final = system.simulate(until=1.0, dt=0.01).get_values('film.Q')[-1]
    assert np.allclose(final, start * np.exp(-1.0), rtol=1e-4)


def test_tensor_relaxes_to_the_fixed_matrix_of_its_edges():
    """Test that a traceless tensor between edges holding one fixed matrix relaxes to it at every site within 1e-6."""
    anchor = np.array([[0.4, 0.3], [0.3, -0.4]])
    space = LatticeSpace((5, 5), boundary={'Q': ('fixed', anchor)})
    final = _film(space, _smooth).prepare().simulate(until=60.0, dt=0.05).get_values('film.Q')[-1]
    assert np.allclose(final, np.tile([0.4, 0.3], (25, 1)), atol=1e-6)


def test_tensor_rejects_a_fixed_edge_of_another_symmetry():
    """Test that a fixed matrix without the symmetry of its tensor raises an error that names the tensor."""
    space = LatticeSpace((4, 4), boundary={'Q': ('fixed', [[0.0, 1.0], [2.0, 0.0]])})
    with pytest.raises(ValidationError, match=r'film\.Q'):
        _film(space, _smooth).prepare()
