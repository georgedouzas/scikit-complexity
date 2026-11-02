"""Test the `_base` module of `skcomplex.core.interactions`."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import GroupElement, LatticeElement
from skcomplex.core.interactions import (
    BaseInteraction,
    CascadeInteraction,
    ChoiceInteraction,
    CouplingInteraction,
    CrossingInteraction,
    EvolutionInteraction,
    GroupingInteraction,
    ImitationInteraction,
    ImpulseInteraction,
    JumpInteraction,
    LifeCycleInteraction,
    LocalRuleInteraction,
    MapInteraction,
    MatchingInteraction,
    MovementInteraction,
    RateInteraction,
    ReactionInteraction,
    ReinforcementInteraction,
    RelaxationInteraction,
    SamplingInteraction,
    SignAdjustmentInteraction,
    TopologyInteraction,
    TransferInteraction,
    TransitionInteraction,
)
from skcomplex.core.laws.utils import neighbors
from skcomplex.core.spaces import ContinuousSpace, ContinuousTime, DiscreteTime, LatticeSpace, VonNeumannNeighborhood
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import SpaceError, ValidationError
from skcomplex.specs import BaseSpec, GreedyMatchingSpec, ParameterSpec, RoleSpec, StateSpec, TopologyRequestSpec

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
    'choice': (
        ChoiceInteraction,
        lambda role: {'state': 'x', 'options': (0.0, 1.0), 'utility': lambda n, option: option},
        True,
    ),
    'sampling': (
        SamplingInteraction,
        lambda role: {'state': 'x', 'energy': lambda n: -getattr(n, role).x * neighbors(getattr(n, role).other.x)},
        True,
    ),
    'reinforcement': (ReinforcementInteraction, lambda role: {'scores': ('x',), 'payoff': lambda n, index: 1.0}, True),
}
AGENTS = {
    'choice': (
        ChoiceInteraction,
        {'state': 'x', 'options': (0.0, 1.0), 'utility': lambda agent, model, option: option},
    ),
    'reinforcement': (ReinforcementInteraction, {'scores': ('x',), 'payoff': lambda agent, model, index: 1.0}),
    'reaction': (ReactionInteraction, {'reactants': {'x': 1}, 'products': {}}),
    'imitation': (ImitationInteraction, {'state': 'x', 'payoff': 'x', 'probability': 0.5}),
}

AGENT_ALIASES = {
    'choice-intensity': ('choice', 'intensity', 2.0),
    'choice-activity': ('choice', 'activity', 0.5),
    'reinforcement-rate': ('reinforcement', 'rate', 0.3),
    'reaction-k': ('reaction', 'k', 0.4),
    'imitation-probability': ('imitation', 'probability', 0.5),
}
SYMBOLIC_ALIASES = {
    'choice-intensity': (ChoiceInteraction, 'intensity', 2.0, RENAMED['choice'][1]('target'), True),
    'choice-activity': (ChoiceInteraction, 'activity', 0.5, RENAMED['choice'][1]('target'), True),
    'reinforcement-rate': (ReinforcementInteraction, 'rate', 0.3, RENAMED['reinforcement'][1]('target'), True),
    'sampling-temperature': (SamplingInteraction, 'temperature', 2.0, RENAMED['sampling'][1]('target'), True),
    'reaction-k': (ReactionInteraction, 'k', 0.4, {'reactants': {'x': 1}, 'products': {}}, False),
    'coupling-lag': (CouplingInteraction, 'lag', 0.3, {'read': 'x', 'write': 'x'}, False),
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


def _get_fixed(interaction):
    """Return a subclass of a general interaction that declares no parameter, so it reads its attributes."""

    class FixedInteraction(interaction):
        """Interaction that declares no parameter."""

        def __init__(self, label, *, target: Annotated[str, RoleSpec()], **arguments):
            super().__init__(label, target=target, **arguments)

    return FixedInteraction


def _get_aliased(interaction):
    """Return a subclass of a general interaction that reads one parameter under the parameter `alias`."""

    class AliasInteraction(interaction):
        """Interaction that reads a renamed parameter as `alias`."""

        def __init__(
            self,
            label,
            *,
            target: Annotated[str, RoleSpec()],
            renamed: str,
            alias: Annotated[float, ParameterSpec()],
            **arguments,
        ):
            super().__init__(label, target=target, **arguments)
            self.renamed = renamed
            self.alias = alias

        def _get_parameter(self, name):
            """Return `alias` for the renamed parameter."""
            return 'alias' if name == self.renamed else name

    return AliasInteraction


def _run_agents(interaction):
    """Return the final values of the state `x` of four agents on a lattice driven by an agent form."""
    agents = GroupElement(
        'nodes',
        element=NodeElement,
        size=4,
        location=[(0, 0), (0, 1), (1, 0), (1, 1)],
        values={'x': np.array([0.0, 1.0, 0.0, 1.0])},
    )
    system = System(elements=[agents], interactions=[interaction], space=LatticeSpace((2, 2)), time=DiscreteTime())
    return Simulation(system, random_state=0).check().prepare().run(until=5).trajectory_.agents('nodes', 5).values['x']


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


@pytest.mark.parametrize('name', AGENTS)
def test_get_parameter_value_reads_the_attributes_of_undeclared_parameters(name):
    """Test that the agent form of a subclass that declares no parameter runs as the plain interaction."""
    interaction, arguments = AGENTS[name]
    plain = _run_agents(interaction(name, target='nodes', form='agent', **arguments))
    fixed = _run_agents(_get_fixed(interaction)(name, target='nodes', form='agent', **arguments))
    np.testing.assert_array_equal(fixed, plain)


@pytest.mark.parametrize('case', AGENT_ALIASES)
def test_get_parameter_value_reads_a_renamed_parameter_in_the_agent_form(case):
    """Test that the agent form of a subclass that reads a parameter as `alias` runs as the plain interaction."""
    name, parameter, value = AGENT_ALIASES[case]
    interaction, arguments = AGENTS[name]
    plain = _run_agents(interaction(name, target='nodes', form='agent', **{**arguments, parameter: value}))
    aliased = _get_aliased(interaction)(name, target='nodes', form='agent', renamed=parameter, alias=value, **arguments)
    np.testing.assert_array_equal(_run_agents(aliased), plain)


@pytest.mark.parametrize('case', SYMBOLIC_ALIASES)
def test_get_parameter_symbol_reads_a_renamed_parameter_in_the_symbolic_form(case):
    """Test that the symbolic form of a subclass that reads a parameter as `alias` gives the plain trajectory."""
    interaction, parameter, value, arguments, discrete = SYMBOLIC_ALIASES[case]
    plain = _run_nodes(interaction(case, target='nodes', **{**arguments, parameter: value}), discrete)
    aliased = _get_aliased(interaction)(case, target='nodes', renamed=parameter, alias=value, **arguments)
    np.testing.assert_allclose(_run_nodes(aliased, discrete), plain)


GRID_ONLY = {
    'transfer': (
        TransferInteraction(
            'trade',
            source='agents',
            target='agents',
            state='x',
            amount=1.0,
            pairing='cell',
            form='agent',
        ),
        'pairing',
    ),
    'movement': (MovementInteraction('walk', target='agents', step='neighbor'), 'step'),
    'grouping': (GroupingInteraction('groups', target='agents', groups='neighborhood', act=print), 'groups'),
    'life cycle': (LifeCycleInteraction('life', target='agents', placement='neighbor'), 'placement'),
    'matching': (
        MatchingInteraction(
            'market',
            requesters='agents',
            providers='agents',
            rule=GreedyMatchingSpec(candidates='neighbors'),
        ),
        'candidates',
    ),
}


@pytest.mark.parametrize('case', list(GRID_ONLY))
def test_prepare_rejects_grid_variants_in_a_continuous_space(case):
    """Test that a variant of cells or nodes raises an error naming its alternative in a continuous space."""
    interaction, name = GRID_ONLY[case]
    with pytest.raises(SpaceError, match=rf'The {name} .* needs the cells of a grid .* Use '):
        interaction.check().prepare(ContinuousSpace(extent=(5.0, 5.0)))
    interaction.prepare(LatticeSpace((5, 5)))


@dataclass(frozen=True)
class FixedSpec(BaseSpec):
    """Variant a subclass defines, holding the value its dispatch method returns."""

    value: float = 1.0


class PruningInteraction(TopologyInteraction):
    """Topology change whose `FixedSpec` action its dispatch method lowers to the removal of edges."""

    def _get_requests(self):
        """Return the removal of an edge for a `FixedSpec` action, or the requests of a named action."""
        if isinstance(self.action, FixedSpec):
            return [TopologyRequestSpec('remove-edge', self.edges, self.toward, self.keep, self.layer, ())]
        return super()._get_requests()


SPEC_VARIANTS = {
    'choice rule': lambda spec: ChoiceInteraction(
        'choice',
        target='members',
        state='s',
        options=(0, 1),
        utility=lambda s, o: o,
        rule=spec,
    ),
    'imitation rule': lambda spec: ImitationInteraction('imitation', target='players', rule=spec),
    'grouping groups': lambda spec: GroupingInteraction('groups', target='voters', groups=spec, act=print),
    'evolution selection': lambda spec: EvolutionInteraction('evolution', target='agents', selection=spec),
    'evolution replacement': lambda spec: EvolutionInteraction('evolution', target='agents', replacement=spec),
    'topology action': lambda spec: PruningInteraction('change', target='nodes', action=spec),
    'cascade mode': lambda spec: CascadeInteraction('cascade', target='sites', state='h', threshold=4, mode=spec),
    'coupling mode': lambda spec: CouplingInteraction('coupling', target='nodes', read='x', write='x', mode=spec),
    'map mode': lambda spec: MapInteraction('map', target='nodes', state='x', function=lambda s: 1, mode=spec),
}


@pytest.mark.parametrize('case', list(SPEC_VARIANTS))
def test_check_accepts_a_spec_for_a_variant_and_rejects_an_unknown_name(case):
    """Test that a variant argument takes a spec a subclass defines, and an unknown name lists the named values."""
    assert SPEC_VARIANTS[case](FixedSpec()).check() is not None
    with pytest.raises(ValidationError, match=r'Unknown .* Use one of .*BaseSpec'):
        SPEC_VARIANTS[case]('unknown').check()


class FixedChoiceInteraction(ChoiceInteraction):
    """Choice that takes the value of a `FixedSpec` rule, a variant its dispatch method adds."""

    def _draw_agent(self, agent, model, intensity):
        """Return the value of a `FixedSpec` rule, or the option the parent draws by its named rule."""
        if isinstance(self.rule, FixedSpec):
            return self.rule.value
        return super()._draw_agent(agent, model, intensity)


def test_dispatch_method_of_a_subclass_runs_its_spec_variant():
    """Test that a choice subclass whose dispatch method handles a spec rule sets every agent to its value."""
    group = GroupElement('members', element=NodeElement, size=3)
    choice = FixedChoiceInteraction(
        'choice',
        target='members',
        state='x',
        options=(0, 1, 2),
        utility=print,
        rule=FixedSpec(2.0),
        form='agent',
    )
    system = System(elements=[group], interactions=[choice], time=DiscreteTime())
    trajectory = Simulation(system, random_state=0).check().prepare().run(until=1).trajectory_
    assert trajectory.agents('members', 1).values['x'].tolist() == [2.0, 2.0, 2.0]


def _run_variant(*interactions, space=None, time=None):
    """Run three members under interactions, on a lattice of three sites when a space is not given."""
    group = GroupElement('nodes', element=NodeElement, size=3)
    system = System(
        elements=[group],
        interactions=list(interactions),
        space=LatticeSpace((3,)) if space is None else space,
        time=DiscreteTime() if time is None else time,
    )
    Simulation(system, random_state=0).check().prepare().run(until=2)


def _choice(form):
    """Return a choice with an unhandled spec rule."""
    utility = (lambda agent, model, option: option) if form == 'agent' else (lambda namespace, option: option)
    return ChoiceInteraction(
        'choice',
        target='nodes',
        state='x',
        options=(0, 1),
        utility=utility,
        rule=FixedSpec(),
        form=form,
    )


def _imitation(form):
    """Return an imitation with an unhandled spec rule."""
    return ImitationInteraction(
        'imitation',
        target='nodes',
        state='x',
        options=(0, 1),
        payoff='x',
        probability=1.0,
        rule=FixedSpec(),
        form=form,
    )


UNHANDLED_VARIANTS = {
    'choice agent': (lambda: _run_variant(_choice('agent')), '_draw_agent'),
    'choice symbolic': (lambda: _run_variant(_choice('symbolic')), '_draw_symbolic'),
    'imitation agent': (lambda: _run_variant(_imitation('agent')), '_choose'),
    'imitation rate': (lambda: _run_variant(_imitation('rate'), time=ContinuousTime()), '_get_jumps'),
    'imitation probability': (lambda: _run_variant(_imitation('probability')), '_get_update'),
    'map agent': (
        lambda: _run_variant(
            MapInteraction(
                'map',
                target='nodes',
                state='x',
                function=lambda agent, model: 1.0,
                mode=FixedSpec(),
                form='agent',
            ),
        ),
        '_apply_mode',
    ),
    'map symbolic': (
        lambda: _run_variant(
            MapInteraction('map', target='nodes', state='x', function=lambda namespace: 1.0, mode=FixedSpec()),
        ),
        '_apply_mode',
    ),
    'grouping': (
        lambda: _run_variant(GroupingInteraction('groups', target='nodes', groups=FixedSpec(), act=print)),
        '_get_groups',
    ),
    'evolution selection': (
        lambda: _run_variant(
            EvolutionInteraction('evolution', target='nodes', strategy='x', fitness=None, selection=FixedSpec()),
        ),
        '_select',
    ),
    'evolution replacement': (
        lambda: _run_variant(
            EvolutionInteraction('evolution', target='nodes', strategy='x', fitness=None, replacement=FixedSpec()),
        ),
        '_breed',
    ),
    'topology': (
        lambda: TopologyInteraction('change', target='nodes', action=FixedSpec()).check(),
        '_get_requests',
    ),
    'cascade': (
        lambda: _run_variant(CascadeInteraction('cascade', target='nodes', state='x', threshold=4, mode=FixedSpec())),
        '_relax',
    ),
    'coupling': (
        lambda: _run_variant(CouplingInteraction('coupling', target='nodes', read='x', write='x', mode=FixedSpec())),
        '_get_reduction',
    ),
}


@pytest.mark.parametrize('case', list(UNHANDLED_VARIANTS))
def test_dispatch_raises_for_a_spec_variant_without_an_override(case):
    """Test that a spec variant no subclass handles raises an error naming the method to override."""
    run, method = UNHANDLED_VARIANTS[case]
    with pytest.raises(ValidationError, match=f'Override `{method}`'):
        run()


def test_check_refuses_a_spec_of_groups_in_the_symbolic_form():
    """Test that the symbolic form of a grouping refuses a spec of groups, since it compiles the named ones."""
    grouping = GroupingInteraction(
        'groups',
        target='nodes',
        groups=FixedSpec(),
        write='x',
        function=lambda s: 1.0,
        form='symbolic',
    )
    with pytest.raises(ValidationError, match='compiles the named groups'):
        grouping.check()
