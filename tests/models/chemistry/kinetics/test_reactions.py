"""Test the `_reactions` module of `skcomplex.models.chemistry.kinetics`."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.solvers import EulerMaruyamaSolver
from skcomplex.models import describe
from skcomplex.models.chemistry.kinetics import (
    BindingInteraction,
    SpeciesElement,
    chemical_langevin_equation,
    chemical_langevin_network,
    michaelis_menten,
)

from .conftest import _run


def test_michaelis_menten_conserves_enzyme_and_substrate():
    """Test that the enzyme reaction conserves the total enzyme and substrate as rate equations and as jumps."""
    for stochastic in (False, True):
        trajectory = _run(michaelis_menten(stochastic=stochastic), until=20.0, dt=1.0)
        enzyme = trajectory.values('E.n') + trajectory.values('C.n')
        substrate = trajectory.values('S.n') + trajectory.values('C.n') + trajectory.values('P.n')
        assert np.allclose(enzyme, 100.0)
        assert np.allclose(substrate, 300.0)
        assert trajectory.values('P.n')[-1] > 100.0
    assert np.array_equal(trajectory.values('P.n'), np.round(trajectory.values('P.n')))


def test_chemical_langevin_equation_keeps_both_states():
    """Test that the Schlögl reaction fluctuates around its low or its high state, depending on its start."""
    low = _run(
        chemical_langevin_equation(x=17.0),
        until=1.0,
        dt=0.05,
        solver=describe('chemical_langevin_equation').defaults['solver'],
    )
    high = _run(
        chemical_langevin_equation(x=113.0),
        until=1.0,
        dt=0.05,
        solver=describe('chemical_langevin_equation').defaults['solver'],
    )
    assert abs(low.values('reactor.x').mean() - 17.0) < 10.0
    assert abs(high.values('reactor.x').mean() - 113.0) < 25.0
    assert low.values('reactor.x').std() > 1.0


@pytest.mark.slow
def test_chemical_langevin_equation_switches_by_noise():
    """Test that the default Schlögl reaction switches between its low and its high state within 200 time units."""
    trajectory = _run(
        chemical_langevin_equation(),
        until=200.0,
        dt=0.05,
        solver=describe('chemical_langevin_equation').defaults['solver'],
    )
    late = trajectory.values('reactor.x')[trajectory.times > 20]
    assert late.min() < 30.0
    assert late.max() > 90.0


def test_langevin_binding_conserves_difference():
    """Test that `A + B -> C` as a chemical Langevin equation keeps `A - B` while `A` fluctuates."""
    system = System(
        elements=[SpeciesElement('A', n=200.0), SpeciesElement('B', n=150.0), SpeciesElement('C')],
        interactions=[BindingInteraction('binding', first='A', second='B', product='C', k=0.001, langevin=True)],
    )
    trajectory = _run(system, 5.0, 0.05, solver=EulerMaruyamaSolver(dt=1e-3))
    difference = trajectory.values('A.n') - trajectory.values('B.n')
    np.testing.assert_allclose(difference, 50.0, rtol=1e-9)
    deterministic = _run(
        System(
            elements=[SpeciesElement('A', n=200.0), SpeciesElement('B', n=150.0), SpeciesElement('C')],
            interactions=[BindingInteraction('binding', first='A', second='B', product='C', k=0.001)],
        ),
        5.0,
        0.05,
    )
    assert np.max(np.abs(trajectory.values('A.n') - deterministic.values('A.n'))) > 0.5


def test_langevin_and_stochastic_reaction_is_rejected():
    """Test that a reaction both stochastic and in Langevin mode raises an error that names it."""
    system = System(
        elements=[SpeciesElement('A', n=10.0), SpeciesElement('B', n=10.0), SpeciesElement('C')],
        interactions=[
            BindingInteraction('binding', first='A', second='B', product='C', stochastic=True, langevin=True),
        ],
    )
    with pytest.raises(ValidationError, match='binding'):
        compile_system(system)


def test_chemical_langevin_network_conserves_totals():
    """Test that the enzyme network under noise keeps `E + C` and `S + C + P` at every frame."""
    defaults = describe('chemical_langevin_network').defaults
    trajectory = _run(chemical_langevin_network(), defaults['until'], defaults['dt'], solver=defaults['solver'])
    enzyme = trajectory.values('E.n') + trajectory.values('C.n')
    substrate = trajectory.values('S.n') + trajectory.values('C.n') + trajectory.values('P.n')
    np.testing.assert_allclose(enzyme, 100.0, rtol=1e-9)
    np.testing.assert_allclose(substrate, 300.0, rtol=1e-9)
    assert np.std(np.diff(trajectory.values('C.n'))) > 0
