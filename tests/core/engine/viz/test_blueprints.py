"""Test the views of the visuals and the default blueprint that `RerunObserver` builds from them."""

from typing import Annotated

import pytest
import rerun.blueprint as rrb

from skcomplex.core.components.elements import Element, FieldElement, GroupElement
from skcomplex.core.components.spaces import ContinuousSpace, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.problems import BaseProblemSpec
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import (
    AgentsVisual,
    BaseVisual,
    ImageVisual,
    PointsVisual,
    RerunObserver,
    ScalarsVisual,
)
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import StateSpec, VisualSpec
from skcomplex.systems.biology.collective_behaviour import BoidsSystem
from skcomplex.systems.biology.common.populations import PopulationElement
from skcomplex.systems.biology.ecology import LotkaVolterraSystem, WolfSheepSystem
from skcomplex.systems.biology.epidemiology import SIRSystem
from skcomplex.systems.biology.evolution import MoranProcessSystem
from skcomplex.systems.chemistry.pattern_formation import GrayScottSystem
from skcomplex.systems.computer_science.automata import ElementaryAutomatonSystem
from skcomplex.systems.mathematics.networks import NetworkDiffusionSystem
from skcomplex.systems.physics.heat import HeatElement
from skcomplex.systems.physics.statistical_mechanics import IdealGasSystem
from skcomplex.systems.social.opinion_dynamics import VoterSystem


def _prepared(system, **settings):
    """Return a Rerun observer prepared on a system or a compiled problem."""
    problem = system if isinstance(system, BaseProblemSpec) else PassCompiler().prepare(system).run().problem_
    return RerunObserver(**settings).prepare(problem)


def default_blueprint(system, **settings):
    """Return the default blueprint of a system or a compiled problem."""
    return _prepared(system, blueprint='default', **settings).blueprint_


def primary_view(system):
    """Return the primary view of a system or a compiled problem."""
    return _prepared(system).views_[0]


def _views(blueprint):
    """Return the views of a blueprint in layout order."""
    found = []
    stack = [blueprint.root_container]
    while stack:
        part = stack.pop(0)
        if isinstance(part, rrb.View):
            found.append(part)
        else:
            stack = list(part.contents) + stack
    return found


def test_views_of_builtin_visuals():
    """Test that the scalars, 3D points, image, and network agents visuals declare their views."""
    lotka = PassCompiler().prepare(LotkaVolterraSystem().prepare().system_).run().problem_
    gas = PassCompiler().prepare(IdealGasSystem(n_molecules=10, n_dim=3).prepare().system_).run().problem_
    gray = PassCompiler().prepare(GrayScottSystem(shape=(16, 16)).prepare().system_).run().problem_
    voters = PassCompiler().prepare(VoterSystem().prepare().system_).run().problem_
    views = [
        ScalarsVisual(path=lotka.elements[0].path).prepare(lotka).views_[0],
        PointsVisual(position='x', states=(), path=gas.elements[0].path).prepare(gas).views_[0],
        ImageVisual(path=gray.elements[0].path).prepare(gray).views_[0],
        AgentsVisual(path=voters.agents[0].path).prepare(voters).views_[0],
    ]
    assert [type(view).__name__ for view in views] == ['TimeSeriesView', 'Spatial3DView', 'Spatial2DView', 'GraphView']
    assert [str(view.origin) for view in views] == ['prey', 'gas', 'cells/u', 'voters']


def test_views_of_user_visual_are_empty_by_default():
    """Test that a visual that declares no views is left out of the layout."""
    problem = PassCompiler().prepare(LotkaVolterraSystem().prepare().system_).run().problem_
    assert BaseVisual(path=problem.elements[0].path).prepare(problem).views_ == []


def test_views_of_points_log_other_states_as_series():
    """Test that the states other than the position get a time series view."""
    problem = PassCompiler().prepare(IdealGasSystem(n_molecules=10).prepare().system_).run().problem_
    views = PointsVisual(position='x', path=problem.elements[0].path).prepare(problem).views_
    assert [type(view).__name__ for view in views] == ['Spatial2DView', 'TimeSeriesView']


def _whole_run_series(blueprint):
    """Return whether every time series view of a blueprint shows the time range of the whole run."""
    series = [view for view in _views(blueprint) if isinstance(view, rrb.TimeSeriesView)]
    return bool(series) and all('TimeAxis' in view.properties for view in series)


def test_default_blueprint_shows_one_dimensional_points_over_whole_run():
    """Test that the time series view of the positions of a one-dimensional gas shows the whole run."""
    blueprint = default_blueprint(IdealGasSystem(n_molecules=10, n_dim=1).prepare().system_)
    assert 'gas/x' in [str(view.origin) for view in _views(blueprint)]
    assert _whole_run_series(blueprint)


def test_default_blueprint_shows_other_states_of_points_over_whole_run():
    """Test that the time series view of the other states of two-dimensional points shows the whole run."""

    class WalkerElement(Element):
        """Walker element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = VisualSpec('points', {'position': 'x'}),
            *,
            x: Annotated[object, StateSpec(vector=True)] = 0.0,
            energy: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x
            self.energy = energy

    blueprint = default_blueprint(
        System(elements=[GroupElement('walkers', element=WalkerElement, size=3)], space=ContinuousSpace(n_dim=2)),
    )
    assert [type(view).__name__ for view in _views(blueprint)] == ['Spatial2DView', 'TimeSeriesView']
    assert _whole_run_series(blueprint)


def test_default_blueprint_of_ode_system():
    """Test that the populations share one time series view, without parameters, collapsed, and following."""
    blueprint = default_blueprint(LotkaVolterraSystem().prepare().system_)
    views = _views(blueprint)
    assert len(views) == 1
    assert isinstance(views[0], rrb.TimeSeriesView)
    assert views[0].contents == ['/prey/**', '/predators/**']
    assert blueprint.collapse_panels
    assert blueprint.time_panel.play_state == 'following'
    assert blueprint.top_panel.state == 'hidden'


def test_default_blueprint_of_field_system():
    """Test that each field state gets a 2D view side by side."""
    blueprint = default_blueprint(GrayScottSystem(shape=(16, 16)).prepare().system_)
    assert isinstance(blueprint.root_container, rrb.Horizontal)
    assert [str(view.origin) for view in _views(blueprint)] == ['cells/u', 'cells/v']


def test_default_blueprint_puts_spatial_views_above_series():
    """Test that agents share one spatial view above the time series of their counts."""
    blueprint = default_blueprint(WolfSheepSystem().prepare().system_)
    assert isinstance(blueprint.root_container, rrb.Vertical)
    spatial, series = _views(blueprint)
    assert spatial.contents == ['/grass/**', '/sheep/**', '/wolves/**']
    assert isinstance(series, rrb.TimeSeriesView)


def test_default_blueprint_splits_series_by_six():
    """Test that a type with more than six series is split over several views."""
    system = System(elements=[PopulationElement(f'p{index}') for index in range(8)])
    names = [str(view.name) for view in _views(default_blueprint(system))]
    assert names == ['PopulationElement 1', 'PopulationElement 2']


def test_default_blueprint_with_parameters():
    """Test that parameters get a time series view on request."""
    views = _views(default_blueprint(LotkaVolterraSystem().prepare().system_, parameters=True))
    assert [str(view.origin) for view in views][-1] == '/'
    assert views[-1].contents == ['/parameters/**']


@pytest.mark.parametrize(
    ('system_class', 'kind', 'origin'),
    [(GrayScottSystem, 'Spatial2DView', 'cells/u'), (SIRSystem, 'TimeSeriesView', '/')],
)
def test_primary_view_is_first_view(system_class, kind, origin):
    """Test that the primary view is the first view of the default blueprint."""
    system = system_class().prepare().system_
    view = primary_view(system)
    assert (type(view).__name__, str(view.origin)) == (kind, origin)
    assert str(_views(default_blueprint(system))[0].origin) == origin


def test_views_are_empty_without_declared_views():
    """Test that a system with no declared views has no views and an empty default blueprint."""
    observer = _prepared(System(), blueprint='default')
    assert observer.views_ == []
    assert _views(observer.blueprint_) == []


@pytest.mark.parametrize('settings', [{'primary': 'cells/u'}, {'parameters': True}, {'blueprint': 'grid'}])
def test_rerun_observer_check_rejects_settings_without_the_default_blueprint(settings):
    """Test that the settings of the default blueprint are rejected without it, as is an unknown blueprint name."""
    with pytest.raises(ValidationError, match='default'):
        RerunObserver(**settings).prepare()


def test_rerun_observer_saves_blueprint_next_to_recording(tmp_path):
    """Test that a saved recording gets its blueprint with the same stem."""
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd', blueprint='default')
    simulate(LotkaVolterraSystem().prepare().system_, until=1, dt=0.1, observers=[observer])
    assert (tmp_path / 'run.rrd').exists()
    assert (tmp_path / 'run.rbl').exists()
    assert isinstance(observer.blueprint_, rrb.Blueprint)


def test_rerun_observer_without_blueprint_writes_recording_only(tmp_path):
    """Test that no blueprint is sent or saved when none is given."""
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd')
    simulate(LotkaVolterraSystem().prepare().system_, until=1, dt=0.1, observers=[observer])
    assert observer.blueprint_ is None
    assert not (tmp_path / 'run.rbl').exists()


def test_rerun_observer_accepts_given_blueprint(tmp_path):
    """Test that a given blueprint is sent unchanged."""
    blueprint = rrb.Blueprint(rrb.TimeSeriesView(origin='/prey'))
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd', blueprint=blueprint)
    simulate(LotkaVolterraSystem().prepare().system_, until=1, dt=0.1, observers=[observer])
    assert observer.blueprint_ is blueprint


def test_rerun_observer_rejects_unknown_blueprint(tmp_path):
    """Test that an unknown blueprint name is rejected."""
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd', blueprint='grid')
    with pytest.raises(ValidationError, match='default'):
        simulate(LotkaVolterraSystem().prepare().system_, until=1, dt=0.1, observers=[observer])


def test_default_blueprint_puts_named_primary_view_first():
    """Test that a view named as primary comes first, with its row above the others."""
    views = _views(default_blueprint(BoidsSystem().prepare().system_, primary='reporters'))
    assert [str(view.name) for view in views] == ['reporters', 'flock']


def test_default_blueprint_rejects_unknown_primary_view():
    """Test that a primary view that matches no view is rejected with the names of the views."""
    with pytest.raises(ValidationError, match='The views are: u, v'):
        default_blueprint(GrayScottSystem(shape=(16, 16)).prepare().system_, primary='cells/w')


def test_primary_view_of_one_dimensional_lattice_is_space_time():
    """Test that a one-dimensional lattice puts its space-time image first."""
    view = primary_view(ElementaryAutomatonSystem().prepare().system_)
    assert type(view).__name__ == 'Spatial2DView'
    assert str(view.origin) == 'cells/space_time/alive'
    assert str(view.name) == 'alive history'


def test_primary_view_of_agents_without_space_is_a_trait_series():
    """Test that agents without a space show the statistics of their colored trait first, not the reporters."""
    problem = PassCompiler().prepare(MoranProcessSystem().prepare().system_).run().problem_
    assert problem.agent_space is None
    assert AgentsVisual(path=problem.agents[0].path).prepare(problem).views_ == []
    view = primary_view(problem)
    assert (type(view).__name__, str(view.name)) == ('TimeSeriesView', 'population trait')


def test_views_of_image_without_history_leave_out_space_time():
    """Test that a history of 0 rows declares no space-time view."""
    problem = PassCompiler().prepare(ElementaryAutomatonSystem().prepare().system_).run().problem_
    views = ImageVisual(history=0, path=problem.elements[0].path).prepare(problem).views_
    assert [str(view.origin) for view in views] == ['cells/alive']


def test_fields_on_networks_are_drawn_as_colored_graphs():
    """Test that a field on a network without a visual is drawn as a colored graph, not one series per node."""
    problem = PassCompiler().prepare(NetworkDiffusionSystem().prepare().system_).run().problem_
    assert problem.elements[0].visual.kind == 'sites'
    assert any(type(view).__name__ == 'GraphView' for view in _views(default_blueprint(problem)))


def test_fields_keep_a_given_visual():
    """Test that a field keeps the visual it is given."""
    field = FieldElement('plate', element=HeatElement, space=LatticeSpace((4, 4)), visual=VisualSpec('sites'))
    assert PassCompiler().prepare(System(elements=[field])).run().problem_.elements[0].visual.kind == 'sites'
