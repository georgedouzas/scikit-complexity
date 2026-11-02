"""Test the `_pair` module of `skcomplex.core.interactions`."""

from types import SimpleNamespace
from typing import Annotated

import pytest

from skcomplex.core.elements import Element, GroupElement, PointElement
from skcomplex.core.interactions import PairInteraction
from skcomplex.core.laws import BoundaryLaw, RateLaw
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.errors import ValidationError
from skcomplex.specs import ParameterSpec, StateSpec, VisualSpec


class PopulationElement(Element):
    """Population without a position."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n


class AttractionInteraction(PairInteraction):
    """Pair interaction that returns a boundary law, which pair interactions do not take."""

    def _get_laws(self, s):
        """Return a boundary law."""
        return [BoundaryLaw(s.first.x_d1, s.first.x_d1)]


class ContactInteraction(PairInteraction):
    """Pair interaction between populations, which have no position."""

    def _get_laws(self, s):
        """Return a rate law."""
        return [RateLaw(s.first.n, s.second.n)]


class RepulsionInteraction(PairInteraction):
    """Pair interaction with a stiffness of its own."""

    def __init__(self, label, first, second, cutoff=1.0, *, position='x', k: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label, first, second, cutoff=cutoff, position=position)
        self.k = k


def test_pair_interaction_stores_its_roles_and_cutoff():
    """Test that a pair interaction keeps its groups, its cutoff, and its position, and its subclass parameters."""
    repulsion = RepulsionInteraction('repulsion', 'gas', 'gas', cutoff=0.5, k=2.0)
    assert repulsion.get_params() == {
        'cutoff': 0.5,
        'first': 'gas',
        'k': 2.0,
        'label': 'repulsion',
        'position': 'x',
        'second': 'gas',
    }
    assert list(RepulsionInteraction._get_specs()) == ['first', 'second', 'cutoff', 'k']
    assert repulsion.check().prepare().run(SimpleNamespace()).laws_ == []


def test_pair_interaction_needs_a_positive_cutoff():
    """Test that the cutoff of a pair interaction has a positive domain."""
    assert 0.5 in RepulsionInteraction._get_specs()['cutoff'].domain
    assert 0.0 not in RepulsionInteraction._get_specs()['cutoff'].domain


def test_pair_interaction_rejects_other_law_kinds():
    """Test that pair interactions take rate laws only."""
    system = System(
        elements=[GroupElement('gas', element=PointElement, size=3)],
        interactions=[AttractionInteraction('pull', first='gas', second='gas')],
        space=ContinuousSpace(n_dim=2),
    )
    with pytest.raises(ValidationError, match='laws other than Rate'):
        compile_system(system)


def test_pair_interaction_needs_vector_position():
    """Test that pair interactions need a vector position state."""
    system = System(
        elements=[GroupElement('herds', element=PopulationElement, size=3)],
        interactions=[ContactInteraction('contact', first='herds', second='herds')],
    )
    with pytest.raises(ValidationError, match="vector state 'x'"):
        compile_system(system)
