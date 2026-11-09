"""Test the `_lower_fields` pass of `skcomplex.core.engine.compile.passes`."""

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, FieldElement, GroupElement
from skcomplex.core.components.interactions import DiffusionInteraction, Interaction
from skcomplex.core.components.laws import BaseLaw, LinearProgramLaw, RateLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.specs import LinearObjectiveSpec, RoleSpec, StateSpec

RATE = 2.0


class ChoiceElement(Element):
    """Choice of a growth rate at a site, a state that a linear program sets."""

    def __init__(self, label: str, *, v: Annotated[float, StateSpec(algebraic=True)] = 0.0) -> None:
        super().__init__(label)
        self.v = v

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseLaw]:
        """Return the program of the choice."""
        return [LinearProgramLaw([namespace.v], [namespace.v <= RATE], [LinearObjectiveSpec(namespace.v)])]


class AmountElement(Element):
    """Amount at a site."""

    def __init__(self, label: str, *, x: Annotated[float, StateSpec()] = 1.0) -> None:
        super().__init__(label)
        self.x = x


class ChosenGrowthInteraction(Interaction):
    """Growth of an amount at the rate its site chooses, the largest allowed."""

    def __init__(
        self,
        label: str,
        *,
        choice: Annotated[str, RoleSpec()] = 'choice',
        amount: Annotated[str, RoleSpec()] = 'amount',
    ) -> None:
        super().__init__(label)
        self.choice = choice
        self.amount = amount

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseLaw]:
        """Return the rate of the amount."""
        return [RateLaw(namespace.amount.x, namespace.choice.v * namespace.amount.x)]


def _system() -> System:
    """Return a uniform field that grows at the rate a program sets and diffuses."""
    space = LatticeSpace((4, 4))
    return System(
        elements=[
            GroupElement('choice', element=ChoiceElement, on='sites', space=space),
            FieldElement('amount', element=AmountElement, space=space, values={'x': np.ones((4, 4))}),
        ],
        interactions=[
            ChosenGrowthInteraction('growth', choice='choice', amount='amount'),
            DiffusionInteraction('spread', target='amount', state='x', coefficient=1.0),
        ],
        space=space,
    )


def test_lower_fields_leaves_a_coefficient_of_a_state_out_of_the_linear_terms():
    """Test that a rate with a coefficient that a state of another element gives is not a constant linear term."""
    terms = PassCompiler().prepare(_system()).run().problem_.linear
    assert [term.chain for term in terms] == [(('laplacian', 0),)]
