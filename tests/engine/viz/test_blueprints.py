"""Test the views of the visuals and the default blueprint that `RerunObserver` builds from them."""

from typing import Annotated

import pytest
import rerun.blueprint as rrb

from skcomplex.core.elements import Element, FieldElement, GroupElement
from skcomplex.core.spaces import ContinuousSpace, LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import (
    AgentsVisual,
    BaseVisual,
    ImageVisual,
    PointsVisual,
    RerunObserver,
    ScalarsVisual,
)
from skcomplex.errors import ValidationError
from skcomplex.models import create
from skcomplex.models.biology.ecology import PopulationElement
from skcomplex.models.physics.heat import HeatElement
from skcomplex.specs import BaseProblemSpec, StateSpec, VisualSpec


def _prepared(system, **settings):
    """Return a Rerun observer prepared on a system or a compiled problem."""
    problem = system if isinstance(system, BaseProblemSpec) else compile_system(system)
    return RerunObserver(**settings).check().prepare(problem)


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
    lotka = compile_system(create('lotka_volterra'))
    gas = compile_system(create('ideal_gas', n_molecules=10, n_dim=3))
    gray = compile_system(create('gray_scott', shape=(16, 16)))
    voters = compile_system(create('voter_model'))
    views = [
        ScalarsVisual(path=lotka.elements[0].path).check().prepare(lotka).views_[0],
        PointsVisual(position='x', states=(), path=gas.elements[0].path).check().prepare(gas).views_[0],
        ImageVisual(path=gray.elements[0].path).check().prepare(gray).views_[0],
        AgentsVisual(path=voters.agents[0].path).check().prepare(voters).views_[0],
    ]
    assert [type(view).__name__ for view in views] == ['TimeSeriesView', 'Spatial3DView', 'Spatial2DView', 'GraphView']
    assert [str(view.origin) for view in views] == ['prey', 'gas', 'cells/u', 'voters']


def test_views_of_user_visual_are_empty_by_default():
    """Test that a visual that declares no views is left out of the layout."""
    problem = compile_system(create('lotka_volterra'))
    assert BaseVisual(path=problem.elements[0].path).check().prepare(problem).views_ == []


def test_views_of_points_log_other_states_as_series():
    """Test that the states other than the position get a time series view."""
    problem = compile_system(create('ideal_gas', n_molecules=10))
    views = PointsVisual(position='x', path=problem.elements[0].path).check().prepare(problem).views_
    assert [type(view).__name__ for view in views] == ['Spatial2DView', 'TimeSeriesView']


def _whole_run_series(blueprint):
    """Return whether every time series view of a blueprint shows the time range of the whole run."""
    series = [view for view in _views(blueprint) if isinstance(view, rrb.TimeSeriesView)]
    return bool(series) and all('TimeAxis' in view.properties for view in series)


def test_default_blueprint_shows_one_dimensional_points_over_whole_run():
    """Test that the time series view of the positions of a one-dimensional gas shows the whole run."""
    blueprint = default_blueprint(create('ideal_gas', n_molecules=10, n_dim=1))
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


def test_default_blueprint_of_ode_model():
    """Test that the populations share one time series view, without parameters, collapsed, and following."""
    blueprint = default_blueprint(create('lotka_volterra'))
    views = _views(blueprint)
    assert len(views) == 1
    assert isinstance(views[0], rrb.TimeSeriesView)
    assert views[0].contents == ['/prey/**', '/predators/**']
    assert blueprint.collapse_panels
    assert blueprint.time_panel.play_state == 'following'
    assert blueprint.top_panel.state == 'hidden'


def test_default_blueprint_of_field_model():
    """Test that each field state gets a 2D view side by side."""
    blueprint = default_blueprint(create('gray_scott', shape=(16, 16)))
    assert isinstance(blueprint.root_container, rrb.Horizontal)
    assert [str(view.origin) for view in _views(blueprint)] == ['cells/u', 'cells/v']


def test_default_blueprint_puts_spatial_views_above_series():
    """Test that agents share one spatial view above the time series of their counts."""
    blueprint = default_blueprint(create('wolf_sheep'))
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
    views = _views(default_blueprint(create('lotka_volterra'), parameters=True))
    assert [str(view.origin) for view in views][-1] == '/'
    assert views[-1].contents == ['/parameters/**']


@pytest.mark.parametrize(
    ('name', 'kind', 'origin'),
    [('gray_scott', 'Spatial2DView', 'cells/u'), ('sir', 'TimeSeriesView', '/')],
)
def test_primary_view_is_first_view(name, kind, origin):
    """Test that the primary view is the first view of the default blueprint."""
    system = create(name)
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
        RerunObserver(**settings).check()


def test_rerun_observer_saves_blueprint_next_to_recording(tmp_path):
    """Test that a saved recording gets its blueprint with the same stem."""
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd', blueprint='default')
    Simulation(create('lotka_volterra'), dt=0.1, observers=[observer]).check().run(until=1)
    assert (tmp_path / 'run.rrd').exists()
    assert (tmp_path / 'run.rbl').exists()
    assert isinstance(observer.blueprint_, rrb.Blueprint)


def test_rerun_observer_without_blueprint_writes_recording_only(tmp_path):
    """Test that no blueprint is sent or saved when none is given."""
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd')
    Simulation(create('lotka_volterra'), dt=0.1, observers=[observer]).check().run(until=1)
    assert observer.blueprint_ is None
    assert not (tmp_path / 'run.rbl').exists()


def test_rerun_observer_accepts_given_blueprint(tmp_path):
    """Test that a given blueprint is sent unchanged."""
    blueprint = rrb.Blueprint(rrb.TimeSeriesView(origin='/prey'))
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd', blueprint=blueprint)
    Simulation(create('lotka_volterra'), dt=0.1, observers=[observer]).check().run(until=1)
    assert observer.blueprint_ is blueprint


def test_rerun_observer_rejects_unknown_blueprint(tmp_path):
    """Test that an unknown blueprint name is rejected."""
    observer = RerunObserver('test', save_path=tmp_path / 'run.rrd', blueprint='grid')
    with pytest.raises(ValidationError, match='default'):
        Simulation(create('lotka_volterra'), dt=0.1, observers=[observer]).check().run(until=1)


def test_default_blueprint_puts_named_primary_view_first():
    """Test that a view named as primary comes first, with its row above the others."""
    views = _views(default_blueprint(create('boids'), primary='reporters'))
    assert [str(view.name) for view in views] == ['reporters', 'flock']


def test_default_blueprint_rejects_unknown_primary_view():
    """Test that a primary view that matches no view is rejected with the names of the views."""
    with pytest.raises(ValidationError, match='The views are: u, v'):
        default_blueprint(create('gray_scott', shape=(16, 16)), primary='cells/w')


def test_primary_view_of_one_dimensional_lattice_is_space_time():
    """Test that a one-dimensional lattice puts its space-time image first."""
    view = primary_view(create('elementary_automaton'))
    assert type(view).__name__ == 'Spatial2DView'
    assert str(view.origin) == 'cells/space_time/alive'
    assert str(view.name) == 'alive history'


def test_primary_view_of_agents_without_space_is_a_trait_series():
    """Test that agents without a space show the statistics of their colored trait first, not the reporters."""
    problem = compile_system(create('moran_process'))
    assert problem.agent_space is None
    assert AgentsVisual(path=problem.agents[0].path).check().prepare(problem).views_ == []
    view = primary_view(problem)
    assert (type(view).__name__, str(view.name)) == ('TimeSeriesView', 'population trait')


def test_views_of_image_without_history_leave_out_space_time():
    """Test that a history of 0 rows declares no space-time view."""
    problem = compile_system(create('elementary_automaton'))
    views = ImageVisual(history=0, path=problem.elements[0].path).check().prepare(problem).views_
    assert [str(view.origin) for view in views] == ['cells/alive']


def test_fields_on_networks_are_drawn_as_colored_graphs():
    """Test that a field on a network without a visual is drawn as a colored graph, not one series per node."""
    problem = compile_system(create('network_diffusion'))
    assert problem.elements[0].visual.kind == 'sites'
    assert any(type(view).__name__ == 'GraphView' for view in _views(default_blueprint(problem)))


def test_fields_keep_a_given_visual():
    """Test that a field keeps the visual it is given."""
    field = FieldElement('plate', element=HeatElement, space=LatticeSpace((4, 4)), visual=VisualSpec('sites'))
    assert compile_system(System(elements=[field])).elements[0].visual.kind == 'sites'
