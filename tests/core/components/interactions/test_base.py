"""Test the `_base` module of `skcomplex.core.components.interactions`."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components import interactions
from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.interactions import (
    BaseInteraction,
    CouplingInteraction,
    CrossingInteraction,
    GradientInteraction,
    GroupingInteraction,
    HyperedgeInteraction,
    Interaction,
    JumpInteraction,
    LocalRuleInteraction,
    MapInteraction,
    NeighborInteraction,
    PairInteraction,
    RateInteraction,
    SamplingInteraction,
    TopologyInteraction,
    TransferInteraction,
    TransitionInteraction,
)
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import (
    BallNeighborhood,
    ContinuousSpace,
    ContinuousTime,
    DiscreteTime,
    LatticeSpace,
)
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.enums import ElementKind
from skcomplex.core.errors import SpaceError, ValidationError
from skcomplex.core.specs import BaseSpec, ParameterSpec, PartnerSpec, RoleSpec, StateSpec, TopologyRequestSpec
from skcomplex.systems.general.agents import (
    BirthDeathInteraction,
    ChoiceInteraction,
    EvolutionInteraction,
    ImitationInteraction,
    MovementInteraction,
)
from skcomplex.systems.general.matching import MatchingInteraction
from skcomplex.systems.general.reactions import ReactionInteraction
from skcomplex.systems.physics.common.avalanches import TopplingInteraction

from ....conftest import NodeElement

RENAMED = {
    'rate': (RateInteraction, lambda role: {'rate': lambda n: -getattr(n, role).x}, False),
    'jump': (JumpInteraction, lambda role: {'changes': {'x': 1.0}, 'rate': lambda n: getattr(n, role).x + 1}, False),
    'crossing': (
        CrossingInteraction,
        lambda role: {'guard': lambda n: n.t - 0.5, 'reset': {'x': lambda n: getattr(n, role).x + 2}},
        False,
    ),
    'map': (MapInteraction, lambda role: {'state': 'x', 'function': lambda n: getattr(n, role).x / 2 + 1}, True),
    'transition': (TransitionInteraction, lambda role: {'state': 'x', 'rate': lambda n: 1.0}, False),
    'local_rule': (
        LocalRuleInteraction,
        lambda role: {
            'state': 'x',
            'rule': lambda *values: 1.0 - values[0],
            'neighborhood': BallNeighborhood(metric='manhattan'),
        },
        True,
    ),
    'choice': (
        ChoiceInteraction,
        lambda role: {'state': 'x', 'options': (0.0, 1.0), 'utility': lambda n, option: option},
        True,
    ),
    'sampling': (
        SamplingInteraction,
        lambda role: {'state': 'x', 'objective': lambda n: -getattr(n, role).x * neighbors(getattr(n, role).other.x)},
        True,
    ),
}
AGENTS = {
    'choice': (
        ChoiceInteraction,
        {'state': 'x', 'options': (0.0, 1.0), 'utility': lambda agent, system, option: option},
    ),
    'reaction': (ReactionInteraction, {'reactants': {'x': 1}, 'products': {}}),
    'imitation': (ImitationInteraction, {'state': 'x', 'payoff': 'x', 'probability': 0.5}),
}
AGENT_ALIASES = {
    'choice-intensity': ('choice', 'intensity', 2.0),
    'choice-activity': ('choice', 'activity', 0.5),
    'reaction-k': ('reaction', 'k', 0.4),
    'imitation-probability': ('imitation', 'probability', 0.5),
}
SYMBOLIC_ALIASES = {
    'choice-intensity': (ChoiceInteraction, 'intensity', 2.0, RENAMED['choice'][1]('target'), True),
    'choice-activity': (ChoiceInteraction, 'activity', 0.5, RENAMED['choice'][1]('target'), True),
    'sampling-intensity': (SamplingInteraction, 'intensity', 2.0, RENAMED['sampling'][1]('target'), True),
    'reaction-k': (ReactionInteraction, 'k', 0.4, {'reactants': {'x': 1}, 'products': {}}, False),
    'coupling-lag': (CouplingInteraction, 'lag', 0.3, {'read': 'x', 'write': 'x'}, False),
}
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
    'life cycle': (BirthDeathInteraction('life', target='agents', placement='neighbor'), 'placement'),
    'matching': (
        MatchingInteraction(
            'market',
            requesters='agents',
            providers='agents',
            rule=PartnerSpec('neighbors'),
        ),
        'candidates',
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
    system = CompositeSystem(
        elements=[agents],
        interactions=[interaction],
        space=LatticeSpace((2, 2)),
        time=DiscreteTime(),
    )
    return simulate(system, until=5, random_state=0).get_agents('nodes', 5).values['x']


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
    nodes = GroupElement(
        'nodes',
        element=NodeElement,
        space=LatticeSpace((2, 2)),
        values={'x': np.array([0.0, 1.0, 0.0, 1.0])},
        on='sites',
    )
    system = CompositeSystem(elements=[nodes], interactions=[interaction], time=DiscreteTime() if discrete else None)
    return simulate(system, until=5 if discrete else 1.0, dt=1 if discrete else 0.1, random_state=0).get_values(
        'nodes.x',
    )


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
    growth = GrowthInteraction('growth', target='rabbits')
    assert growth.prepare() is growth
    assert growth.run(SimpleNamespace()).laws_ == []


def test_state_on_an_interaction_is_rejected():
    """Test that checking an interaction that declares a state raises an error that suggests a group on edges."""
    with pytest.raises(ValidationError, match='on="edges"'):
        WeightedInteraction('w').prepare()


def test_adaptive_parameter_on_an_interaction_is_rejected():
    """Test that checking an interaction that declares an adaptive parameter raises an error that names it."""
    with pytest.raises(ValidationError, match='adaptive parameters rate'):
        PolicyInteraction('policy').prepare()


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


@pytest.mark.parametrize('case', list(GRID_ONLY))
def test_prepare_rejects_grid_variants_in_a_continuous_space(case):
    """Test that a variant of cells or nodes raises an error naming its alternative in a continuous space."""
    interaction, name = GRID_ONLY[case]
    with pytest.raises(SpaceError, match=rf'The {name} .* needs the cells of a grid .* Use '):
        interaction.prepare(ContinuousSpace(extent=(5.0, 5.0)))
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
    'cascade mode': lambda spec: TopplingInteraction('cascade', target='sites', state='h', threshold=4, mode=spec),
    'coupling mode': lambda spec: CouplingInteraction('coupling', target='nodes', read='x', write='x', mode=spec),
    'map mode': lambda spec: MapInteraction('map', target='nodes', state='x', function=lambda s: 1, mode=spec),
}


@pytest.mark.parametrize('case', list(SPEC_VARIANTS))
def test_check_accepts_a_spec_for_a_variant_and_rejects_an_unknown_name(case):
    """Test that a variant argument takes a spec a subclass defines, and an unknown name lists the named values."""
    assert SPEC_VARIANTS[case](FixedSpec()).prepare() is not None
    match = r'Unknown .* Use one of .*, or a spec that the component handles\.'
    with pytest.raises(ValidationError, match=match) as error:
        SPEC_VARIANTS[case]('unknown').prepare()
    assert 'BaseSpec' not in str(error.value)


def test_unknown_mode_of_a_map_names_only_the_accepted_values():
    """Test that an unknown mode of a map names the value given and the accepted names, and no base class."""
    with pytest.raises(ValidationError) as error:
        MapInteraction('m', target='a', state='x', function=lambda s: s.target.x, mode='add').prepare()
    message = str(error.value)
    expected = "Unknown mode 'add' of 'm'. Use one of 'replace', 'increment', or a spec that the component handles."
    assert message == expected


class FixedChoiceInteraction(ChoiceInteraction):
    """Choice that takes the value of a `FixedSpec` rule, a variant its dispatch method adds."""

    def _draw_agent(self, agent, system, intensity):
        """Return the value of a `FixedSpec` rule, or the option the parent draws by its named rule."""
        if isinstance(self.rule, FixedSpec):
            return self.rule.value
        return super()._draw_agent(agent, system, intensity)


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
    system = CompositeSystem(elements=[group], interactions=[choice], time=DiscreteTime())
    trajectory = simulate(system, until=1, random_state=0)
    assert trajectory.get_agents('members', 1).values['x'].tolist() == [2.0, 2.0, 2.0]


def _run_variant(*interactions, space=None, time=None):
    """Run three members under interactions, on a lattice of three sites when a space is not given."""
    group = GroupElement('nodes', element=NodeElement, size=3)
    system = CompositeSystem(
        elements=[group],
        interactions=list(interactions),
        space=LatticeSpace((3,)) if space is None else space,
        time=DiscreteTime() if time is None else time,
    )
    simulate(system, until=2, random_state=0)


def _choice(form):
    """Return a choice with an unhandled spec rule."""
    utility = (lambda agent, system, option: option) if form == 'agent' else (lambda namespace, option: option)
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
                function=lambda agent, system: 1.0,
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
        lambda: TopologyInteraction('change', target='nodes', action=FixedSpec()).prepare(),
        '_get_requests',
    ),
    'cascade': (
        lambda: _run_variant(TopplingInteraction('cascade', target='nodes', state='x', threshold=4, mode=FixedSpec())),
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
        grouping.prepare()


def _interaction_classes() -> list[type[BaseInteraction]]:
    """Return the public interaction classes of the core."""
    named = [getattr(interactions, name) for name in interactions.__all__]
    return [item for item in named if isinstance(item, type) and issubclass(item, BaseInteraction)]


def test_get_requirements_names_the_state_of_an_argument():
    """Test that the requirements name the role, its element kinds, the state its argument holds, and the laws."""
    rate = RateInteraction('growth', target='rabbits', state='n', rate=1.0)
    needs = rate.get_requirements()
    assert needs.roles['target'].states == ('n',)
    assert needs.roles['target'].accepts(ElementKind.ELEMENT)
    assert needs.roles['process'].noise
    assert RateLaw in needs.laws
    assert needs.time == (ContinuousTime,)


def test_get_requirements_reads_names_from_sequences_and_mappings():
    """Test that an argument that holds several names, in a sequence or as the keys of a mapping, gives them all."""
    gradient = GradientInteraction('flow', target='a', states=('x', 'y'), potential=lambda s: 0.0)
    assert gradient.get_requirements().roles['target'].states == ('x', 'y')
    reaction = ReactionInteraction('r', target='a', reactants={'S': 1, ('b', 'n'): 1}, products={'P': 1})
    assert reaction.get_requirements().roles['target'].states == ('S', 'P')


def test_get_requirements_leaves_unknown_the_laws_a_subclass_writes():
    """Test that a subclass that writes its own laws lists none, since only running it shows them."""

    class OwnLawsInteraction(RateInteraction):
        """Interaction that returns its own laws."""

        def _get_laws(self, namespace):
            """Return no law."""
            return []

    needs = OwnLawsInteraction('own', target='a').get_requirements()
    assert needs.laws == ()
    assert needs.time == ()


def test_get_requirements_accepts_any_space_when_a_law_does():
    """Test that laws that include one running in any space state no space, and laws that all need one state them."""
    assert RateInteraction('rate', target='a').get_requirements().space == ()
    sampling = SamplingInteraction('sample', target='a', state='s', objective=lambda s: 0.0)
    assert LatticeSpace in sampling.get_requirements().space


def test_every_core_interaction_declares_its_roles():
    """Test that every role of every interaction of the core names the kinds of element it accepts."""
    found = _interaction_classes()
    assert {kind.__name__ for kind in found} == set(interactions.__all__) - {'OPTIMIZERS', 'SCHEDULES'}
    for kind in found:
        for name, spec in kind._get_specs().items():
            if isinstance(spec, RoleSpec) and not spec.noise:
                assert spec.elements, f'{kind.__name__}.{name}'


def test_every_core_interaction_with_laws_lists_them():
    """Test that every interaction of the core lists its laws, apart from the bases a subclass writes laws for."""
    bases = {BaseInteraction, Interaction, NeighborInteraction, PairInteraction, HyperedgeInteraction}
    for kind in set(_interaction_classes()) - bases:
        assert kind.law_classes, kind.__name__


@pytest.mark.parametrize(
    ('interaction', 'laws', 'randomness'),
    [
        (RateInteraction('growth', target='a', rate=1.0), {'rate'}, 'deterministic'),
        (RateInteraction('growth', target='a', rate=1.0, noise=0.1), {'rate', 'noise'}, 'stochastic'),
        (ReactionInteraction('decay', target='a', reactants={'x': 1}), {'rate'}, 'deterministic'),
        (ReactionInteraction('decay', target='a', reactants={'x': 1}, mode='jump'), {'jump'}, 'stochastic'),
        (
            ReactionInteraction('decay', target='a', reactants={'x': 1}, mode='langevin'),
            {'rate', 'noise'},
            'stochastic',
        ),
        (JumpInteraction('flip', target='a', changes={'x': 1.0}, probability=lambda s: 0.5), {'update'}, 'stochastic'),
        (MapInteraction('map', target='a', state='x', function=lambda s: s.target.x), {'update'}, 'deterministic'),
    ],
)
def test_interaction_tags_follow_its_arguments(interaction, laws, randomness):
    """Test that the law kinds and the randomness of an interaction follow the arguments that choose its laws."""
    tags = interaction.get_tags()
    assert (tags.laws, tags.randomness) == (laws, randomness)


def test_interaction_tags_give_the_times_and_structure_of_the_chosen_laws():
    """Test that an interaction gives the times its chosen laws run in, and how they change the structure."""
    assert MapInteraction('map', target='a', state='x', function=lambda s: s.target.x).get_tags().time == {'discrete'}
    assert RateInteraction('growth', target='a', rate=1.0).get_tags().time == {'continuous'}
    assert TopologyInteraction('rewire').get_tags().structure == {'changing topology'}


def test_interaction_that_writes_unlisted_laws_has_unknown_law_kinds():
    """Test that a subclass that writes its laws without listing them leaves its law kinds and randomness unknown."""

    class CustomRateInteraction(RateInteraction):
        """Rate interaction with laws of its own."""

        def _get_laws(self, namespace):
            return []

    tags = CustomRateInteraction('growth', target='a', rate=1.0).get_tags()
    assert (tags.laws, tags.randomness) == (None, None)
    assert Interaction('free').get_tags().laws is None
