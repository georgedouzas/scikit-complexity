"""Test the `types` module of `skcomplex.core.components`."""

import typing
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import PartnerSpec, UpdateLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.types import AgentViewProtocol, Process, SystemViewProtocol, Target
from skcomplex.core.systems import System
from skcomplex.systems.general.matching import RandomSearchSpec
from skcomplex.systems.general.matching.types import MatchingRule


def test_targets_are_symbolic_expressions_or_vectors():
    """Test that a law targets a symbolic expression or a vector of them."""
    assert typing.get_args(Target.__value__) == (Expression, FreeModuleElement)


def test_processes_may_be_nothing():
    """Test that a process may be nothing."""
    assert type(None) in typing.get_args(Process.__value__)


def test_agent_views_have_an_id_a_location_and_a_life():
    """Test that an object with the id, location, and moves of an agent is an agent view, and an id alone is not."""
    names = (
        'get_neighbor_cells',
        'get_neighbors',
        'get_agents_here',
        'get_regions_here',
        'get_displacement_to',
        'get_distance_to',
        'move_to',
        'remove',
        'create',
        'get_referenced',
        'get_lags',
    )
    view = SimpleNamespace(
        id=1,
        group='walkers',
        group_label='walkers',
        location=(0, 0),
        location_key=(0, 0),
        alive=True,
        **dict.fromkeys(names, print),
    )
    assert isinstance(view, AgentViewProtocol)
    assert not isinstance(SimpleNamespace(id=1), AgentViewProtocol)


def test_matching_rules_are_the_matching_specs():
    """Test that a matching rule is one of the specs of the matching algorithms."""
    rules = typing.get_args(MatchingRule.__value__)
    assert RandomSearchSpec in rules
    assert PartnerSpec in rules
    assert len(rules) == 8


def test_views_of_the_agent_stepper_offer_every_member_of_the_protocols() -> None:
    """Test that a rule typed by the protocols reads the members the system and agent views offer."""
    readings: list[tuple[object, ...]] = []

    def read(system: SystemViewProtocol) -> None:
        """Record what the system view and the view of the first walker offer."""
        walker = system.get_agents('walkers')[0]
        readings.append(
            (
                isinstance(system, SystemViewProtocol),
                isinstance(walker, AgentViewProtocol),
                system.round,
                system.count('walkers'),
                walker.group_label,
                walker.get_displacement_to(walker).tolist(),
                dict(system.parameters),
                dict(system.inputs),
                system.get_random_empty_cell() not in {(0, 0), (1, 1)},
            ),
        )

    walkers = Element('walkers', values={'x': [0, 1], 'y': [0, 1]}, size=2)
    census = Interaction('census', roles={'walkers': 'walkers'}, laws=lambda s: [UpdateLaw(s, read)])
    system = System(elements=[walkers], interactions=[census], space=LatticeSpace((3, 3)))
    system.prepare().simulate(until=1, random_state=0)
    assert readings == [(True, True, 0, 2, 'walkers', [0.0, 0.0], {}, {}, True)]
