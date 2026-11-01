"""Test the `_base` module of `skcomplex.core.interactions`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import LatticeElement
from skcomplex.core.interactions import (
    BaseInteraction,
    CrossingInteraction,
    ImpulseInteraction,
    JumpInteraction,
    LocalRuleInteraction,
    MapInteraction,
    RateInteraction,
    RelaxationInteraction,
    SignAdjustmentInteraction,
    TransitionInteraction,
)
from skcomplex.core.spaces import DiscreteTime, LatticeSpace, VonNeumannNeighborhood
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.specs import ParameterSpec, RoleSpec, StateSpec

from ...conftest import NodeElement

RENAMED = {
    'rate': (RateInteraction, lambda role: {'rate': lambda n: -getattr(n, role).x}, False),
    'relaxation': (RelaxationInteraction, lambda role: {'goal': 2.0}, False),
    'jump': (JumpInteraction, lambda role: {'changes': {'x': 1.0}, 'rate': lambda n: getattr(n, role).x + 1}, False),
    'crossing': (
        CrossingInteraction,
        lambda role: {'guard': lambda n: n.t - 0.5, 'reset': {'x': lambda n: getattr(n, role).x + 2}},
        False,
    ),
    'impulse': (
        ImpulseInteraction,
        lambda role: {'reset': {'x': lambda n: getattr(n, role).x + 1}, 'times': (0.5,)},
        False,
    ),
    'map': (MapInteraction, lambda role: {'state': 'x', 'function': lambda n: getattr(n, role).x / 2 + 1}, True),
    'sign_adjustment': (
        SignAdjustmentInteraction,
        lambda role: {'state': 'x', 'signal': lambda n: 1.0, 'mode': 'additive'},
        True,
    ),
    'transition': (TransitionInteraction, lambda role: {'state': 'x', 'rate': lambda n: 1.0}, False),
    'local_rule': (
        LocalRuleInteraction,
        lambda role: {'state': 'x', 'rule': lambda *values: 1.0 - values[0], 'neighborhood': VonNeumannNeighborhood()},
        True,
    ),
}


class GrowthInteraction(BaseInteraction):
    """Interaction with a role and a parameter."""

    def __init__(self, label, target: Annotated[str, RoleSpec()], r: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label)
        self.target = target
        self.r = r

    def _get_laws(self, namespace):
        """Return no laws."""
        return []


def _get_renamed(interaction):
    """Return a subclass of a general interaction that binds its target as `stock` and declares no parameter."""

    class StockInteraction(interaction):
        """Interaction that binds its target as `stock`."""

        def __init__(self, label, *, stock: Annotated[str, RoleSpec()], **arguments):
            super().__init__(label, target=stock, **arguments)
            self.stock = stock

        def _get_role(self, role):
            """Return `stock` for the target."""
            return {'target': 'stock'}.get(role, role)

    return StockInteraction


def _run_nodes(interaction, discrete):
    """Return the trajectory of the state `x` of four lattice nodes driven by an interaction."""
    nodes = LatticeElement(
        'nodes',
        element=NodeElement,
        space=LatticeSpace((2, 2)),
        values={'x': np.array([0.0, 1.0, 0.0, 1.0])},
    )
    system = System(elements=[nodes], interactions=[interaction], time=DiscreteTime() if discrete else None)
    simulation = Simulation(system, dt=1 if discrete else 0.1, random_state=0)
    return simulation.check().prepare().run(until=5 if discrete else 1.0).trajectory_.values('nodes.x')


class WeightedInteraction(BaseInteraction):
    """Interaction that declares a state, which interactions do not hold."""

    def __init__(self, label, weight: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.weight = weight

    def _get_laws(self, namespace):
        """Return no laws."""
        return []


class PolicyInteraction(BaseInteraction):
    """Interaction that declares an adaptive parameter, which interactions do not hold."""

    def __init__(self, label, rate: Annotated[float, ParameterSpec(adaptive=True)] = 1.0):
        super().__init__(label)
        self.rate = rate

    def _get_laws(self, namespace):
        """Return no laws."""
        return []


def test_interaction_stores_its_roles_and_parameters():
    """Test that an interaction stores its arguments, which scikit-learn lists."""
    assert GrowthInteraction('growth', target='rabbits', r=0.5).get_params() == {
        'label': 'growth',
        'r': 0.5,
        'target': 'rabbits',
    }


def test_interaction_prepares_into_itself():
    """Test that a checked interaction has nothing to prepare and runs into its laws."""
    growth = GrowthInteraction('growth', target='rabbits').check()
    assert growth.prepare() is growth
    assert growth.run(SimpleNamespace()).laws_ == []


def test_state_on_an_interaction_is_rejected():
    """Test that checking an interaction that declares a state raises an error that suggests a group on edges."""
    with pytest.raises(ValidationError, match='on="edges"'):
        WeightedInteraction('w').check()


def test_adaptive_parameter_on_an_interaction_is_rejected():
    """Test that checking an interaction that declares an adaptive parameter raises an error that names it."""
    with pytest.raises(ValidationError, match='adaptive parameters rate'):
        PolicyInteraction('policy').check()


@pytest.mark.parametrize('name', RENAMED)
def test_get_member_reads_a_renamed_target_in_the_symbolic_form(name):
    """Test that a subclass that binds its target as `stock` and fixes its parameters gives the plain trajectory."""
    interaction, arguments, discrete = RENAMED[name]
    plain = _run_nodes(interaction(name, target='nodes', **arguments('target')), discrete)
    renamed = _run_nodes(_get_renamed(interaction)(name, stock='nodes', **arguments('stock')), discrete)
    np.testing.assert_allclose(renamed, plain)
