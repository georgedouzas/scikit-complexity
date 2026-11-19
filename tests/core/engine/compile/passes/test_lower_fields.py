"""Test the `_lower_fields` pass of `skcomplex.core.engine.compile.passes`."""

from types import SimpleNamespace

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ConstraintLaw, ObjectiveLaw, RateLaw
from skcomplex.core.components.laws.utils import laplacian
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.systems import System
from skcomplex.core.types import Role

RATE = 2.0


class ChosenGrowthInteraction(Interaction):
    """Growth of an amount at the rate its site chooses, the largest allowed."""

    def __init__(self, label: str, *, choice: Role = 'choice', amount: Role = 'amount') -> None:
        super().__init__(label)
        self.choice = choice
        self.amount = amount

    def _prepare(self, context=None):
        """Set the program of the choice and the rate of the amount."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            v, x = context.choice.v, context.amount.x
            self.laws_ = [ObjectiveLaw([-v], [v]), ConstraintLaw(v <= RATE), RateLaw(x, v * x)]


def _system() -> System:
    """Return a uniform field that grows at the rate a program sets and diffuses."""
    space = LatticeSpace((4, 4))
    spread = Interaction(
        'spread',
        roles={'target': 'amount'},
        laws=lambda s: [RateLaw(s.target.x, laplacian(s.target.x))],
    )
    return System(
        elements=[
            Element('choice', values={'v': 0.0}, space=space),
            Element('amount', values={'x': np.ones((4, 4))}, space=space),
        ],
        interactions=[ChosenGrowthInteraction('growth'), spread],
        space=space,
    )


def test_lower_fields_leaves_a_coefficient_of_a_state_out_of_the_linear_terms():
    """Test that a rate with a coefficient that a state of another element gives is not a constant linear term."""
    terms = _system().prepare()._problem.linear
    assert [term.chain for term in terms] == [(('laplacian', 0),)]
