"""Test the `_comets` module of `skcomplex.models.biology.ecology`."""

import inspect

import numpy as np
import pytest

from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.engine.solvers import IMEXEulerSolver
from skcomplex.models import create, describe, run
from skcomplex.models.biology.ecology import comets
from skcomplex.specs import RunStateSpec

STEP = 0.1
INOCULUM_PER_LEVEL = 3.0
FAR_DISTANCE = 15


def _field(trajectory: Trajectory, name: str) -> np.ndarray:
    """Return a state of a square lattice with one row per frame."""
    values = trajectory.values(name)
    side = round(np.sqrt(np.reshape(values, (trajectory.times.size, -1)).shape[1]))
    return np.reshape(values, (trajectory.times.size, side, side))


def _radii(biomass: np.ndarray) -> np.ndarray:
    """Return the radius of the area of the boxes above a third of the inoculum, the level of Harcombe et al."""
    return np.sqrt(np.count_nonzero(biomass > biomass[0].max() / INOCULUM_PER_LEVEL, axis=(1, 2)) / np.pi)


def _asymmetry(biomass: np.ndarray) -> float:
    """Return the largest difference from the mirror images and the transpose about the central site of a torus."""
    centre = biomass.shape[1] // 2
    centred = np.roll(biomass, (-centre, -centre), axis=(1, 2))
    mirrors = (
        np.roll(np.flip(centred, axis=1), 1, axis=1),
        np.roll(np.flip(centred, axis=2), 1, axis=2),
        np.swapaxes(centred, 1, 2),
    )
    return float(max(np.abs(centred - mirror).max() for mirror in mirrors))


def _taken_up(trajectory: Trajectory) -> float:
    """Return the sum over the steps of the uptake flux times the biomass of every site, per step length."""
    biomass = _field(trajectory, 'biomass.x')
    exchange = _field(trajectory, 'cells.v_EX_glc__D_e')
    return float(-np.sum(exchange[:-1] * biomass[:-1] * np.diff(trajectory.times)[:, None, None]))


def _run_small(shape: tuple[int, int], until: float, **parameters: float) -> Trajectory:
    """Return a run of the published scheme and step on another lattice."""
    system = create('comets', shape=shape, **parameters)
    return Simulation(system, solver=IMEXEulerSolver(dt=STEP), dt=STEP).check().prepare().run(until=until).trajectory_


@pytest.fixture(scope='module')
def published() -> Trajectory:
    """Return the default run of the model."""
    return run('comets').trajectory_


def test_comets_defaults_are_the_published_values():
    """Test the defaults against Table 1 and the Experimental Procedures of Harcombe et al. (2014)."""
    parameters = inspect.signature(comets).parameters
    published_values = {
        'shape': (50, 50),
        'box': 0.02,
        'inoculum': 3e-7,
        'glucose': 5.0,
        'plate_glucose': 0.0088,
        'biomass_diffusion': 3e-10,
        'glucose_diffusion': 5e-6,
        'vmax': 10.0,
        'km': 0.01,
        'interval': 0.1,
    }
    assert {name: parameters[name].default for name in published_values} == published_values
    assert 'threshold' not in parameters
    assert describe('comets').defaults['dt'] == 0.1
    assert describe('comets').defaults['solver'].dt == 0.1


def test_comets_steps_are_the_published_diffusion_numbers():
    """Test that the coefficients come to about 45 and 0.0027 boxes^2/h, and that only diffusion is the linear part."""
    terms = {term.target.entry.name: term for term in compile_system(comets(shape=(3, 3))).linear}
    assert len(compile_system(comets(shape=(3, 3))).linear) == 2
    assert float(terms['glucose.c'].coefficient) / 0.02**2 == pytest.approx(45.0)
    assert float(terms['biomass.x'].coefficient) / 0.02**2 == pytest.approx(0.0027)


def test_comets_inoculum_is_the_published_mass_at_the_centre():
    """Test that the central box holds 3e-7 g, with the volume under a box from the glucose of the medium."""
    biomass = next(item for item in comets(shape=(5, 5)).elements if item.label == 'biomass').values['x']
    glucose = next(item for item in comets(shape=(5, 5)).elements if item.label == 'glucose').values['c']
    volume = 0.0088 * 0.02**2 / glucose[0, 0]
    assert glucose[0, 0] == pytest.approx(27.753, abs=1e-3)
    assert biomass[2, 2] * volume == pytest.approx(3e-7)
    assert np.count_nonzero(biomass) == 1


def test_comets_step_is_growth_and_then_diffusion():
    """Test one step against the update rules of Harcombe et al. followed by one implicit diffusion solve."""
    shape = (5, 5)
    trajectory = _run_small(shape, STEP)
    biomass = _field(trajectory, 'biomass.x')
    growth = _field(trajectory, 'cells.v_Biomass_Ecoli_core')[0, 2, 2]
    assert growth == pytest.approx(0.8739, rel=0.01)
    assert biomass[1].sum() == pytest.approx(biomass[0].sum() * (1 + growth * STEP), rel=1e-9)


def test_comets_balances_glucose_on_a_lattice_that_runs_out():
    """Test the mass balance, also after the glucose runs out, on a closed lattice."""
    trajectory = _run_small((5, 5), 12.0)
    glucose = _field(trajectory, 'glucose.c')
    assert glucose[-1].sum() < 0.01 * glucose[0].sum()
    assert glucose[0].sum() - glucose[-1].sum() == pytest.approx(_taken_up(trajectory), rel=0.01)


def test_comets_stays_finite_and_non_negative_when_glucose_runs_out():
    """Test every frame at the published step and coefficients, through the depletion of the glucose."""
    trajectory = _run_small((5, 5), 30.0)
    for name in ('biomass.x', 'glucose.c'):
        values = _field(trajectory, name)
        assert np.all(np.isfinite(values))
        assert values.min() >= -1e-9


def test_comets_zeroes_the_fluxes_of_a_site_whose_program_has_no_solution():
    """Test that a depleted lattice stops growing instead of raising an error."""
    trajectory = _run_small((5, 5), 30.0)
    assert np.all(_field(trajectory, 'cells.v_Biomass_Ecoli_core')[-1] == 0.0)
    assert np.all(_field(trajectory, 'cells.v_EX_glc__D_e')[-1] == 0.0)


def test_comets_is_symmetric_on_a_small_lattice():
    """Test the symmetry about the centre on an odd lattice."""
    biomass = _field(_run_small((11, 11), 8.0), 'biomass.x')
    assert _asymmetry(biomass) < 1e-6


def test_comets_gives_zero_fluxes_to_the_sites_without_biomass():
    """Test that the first frame has fluxes at the inoculated site alone."""
    trajectory = _run_small((5, 5), STEP)
    exchange = _field(trajectory, 'cells.v_EX_glc__D_e')[0]
    assert exchange[2, 2] < 0.0
    assert np.count_nonzero(exchange) == 1


def test_comets_program_solves_only_the_sites_with_biomass():
    """Test that one evaluation solves the one inoculated site."""
    problem = compile_system(create('comets', shape=(5, 5)))
    kernel = problem.programs[0]
    kernel.run(RunStateSpec(0.0, problem.initial_state, problem.initial_parameters))
    assert kernel.solves_ == 1


@pytest.mark.slow
def test_comets_default_run_balances_glucose(published: Trajectory):
    """Test the mass balance of the default run within 1%."""
    glucose = _field(published, 'glucose.c')
    assert glucose[0].sum() - glucose[-1].sum() == pytest.approx(_taken_up(published), rel=0.01)


@pytest.mark.slow
def test_comets_default_run_stays_finite_and_non_negative(published: Trajectory):
    """Test every frame of the default run."""
    for name in ('biomass.x', 'glucose.c'):
        values = _field(published, name)
        assert np.all(np.isfinite(values))
        assert values.min() >= -1e-9


@pytest.mark.slow
def test_comets_default_run_is_symmetric_and_its_radius_grows(published: Trajectory):
    """Test the symmetry about the centre at every frame and the growth of the radius over the run."""
    biomass = _field(published, 'biomass.x')
    radii = _radii(biomass)
    assert _asymmetry(biomass) < 1e-6
    assert np.all(np.diff(radii) >= 0.0)
    assert radii[-1] > 4 * radii[0]


@pytest.mark.slow
def test_comets_default_run_holds_its_budget_and_its_caption(published: Trajectory):
    """Test the frames and every claim of the caption of the gallery example against the data."""
    times = published.times
    biomass = _field(published, 'biomass.x')
    glucose = _field(published, 'glucose.c')
    radii = _radii(biomass)
    late = times > times[-1] / 3
    slope = np.polyfit(times[late], radii[late], 1)[0]
    speeds = np.diff(radii[late]) / np.diff(times[late])
    assert times.size <= 300
    assert slope > 0
    assert abs(np.mean(speeds[len(speeds) // 2 :]) - slope) < 0.3 * slope
    terms = {term.target.entry.name: term for term in compile_system(comets(shape=(3, 3))).linear}
    assert float(terms['glucose.c'].coefficient) / float(terms['biomass.x'].coefficient) == pytest.approx(
        17000,
        rel=0.03,
    )
    assert glucose[-1, 25, 25] < 0.5 * glucose[0, 25, 25]
    far = glucose[-1, 25, 25 + FAR_DISTANCE]
    assert biomass[-1, 25, 25 + FAR_DISTANCE] < biomass[0].max() / INOCULUM_PER_LEVEL
    assert far < 0.95 * glucose[0, 25, 25 + FAR_DISTANCE]
