"""Test the `_tensor` module of `skcomplex.core.components.elements`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, Tensor
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import Role
from skcomplex.core.systems import System


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
