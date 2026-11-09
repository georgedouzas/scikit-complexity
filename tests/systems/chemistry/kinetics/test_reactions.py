"""Test the `_reactions` module of `skcomplex.systems.chemistry.kinetics`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import NonnegativeElement
from skcomplex.core.components.interactions import ReactionInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.errors import ValidationError
from skcomplex.systems.chemistry.kinetics import (
    ChemicalLangevinEquationSystem,
    ChemicalLangevinNetworkSystem,
    MichaelisMentenSystem,
)

from .conftest import _run


def test_michaelis_menten_conserves_enzyme_and_substrate():
    """Test that the enzyme reaction conserves the total enzyme and substrate as rate equations and as jumps."""
    for stochastic in (False, True):
        trajectory = _run(MichaelisMentenSystem(stochastic=stochastic), until=20.0, dt=1.0)
        enzyme = trajectory.values('E.x') + trajectory.values('C.x')
        substrate = trajectory.values('S.x') + trajectory.values('C.x') + trajectory.values('P.x')
        assert np.allclose(enzyme, 100.0)
        assert np.allclose(substrate, 300.0)
        assert trajectory.values('P.x')[-1] > 100.0
    assert np.array_equal(trajectory.values('P.x'), np.round(trajectory.values('P.x')))


def test_chemical_langevin_equation_keeps_both_states():
    """Test that the Schlögl reaction fluctuates around its low or its high state, depending on its start."""
    low = _run(ChemicalLangevinEquationSystem(x=17.0), until=1.0, dt=0.05)
    high = _run(ChemicalLangevinEquationSystem(x=113.0), until=1.0, dt=0.05)
    assert abs(low.values('reactor.x').mean() - 17.0) < 10.0
    assert abs(high.values('reactor.x').mean() - 113.0) < 25.0
    assert low.values('reactor.x').std() > 1.0


@pytest.mark.slow
def test_chemical_langevin_equation_switches_by_noise():
    """Test that the default Schlögl reaction switches between its low and its high state within 200 time units."""
    trajectory = simulate(ChemicalLangevinEquationSystem(), random_state=0)
    late = trajectory.values('reactor.x')[trajectory.times > 20]
    assert late.min() < 30.0
    assert late.max() > 90.0


def test_langevin_binding_conserves_difference():
    """Test that `A + B -> C` as a chemical Langevin equation keeps `A - B` while `A` fluctuates."""
    species = [NonnegativeElement('A', x=200.0), NonnegativeElement('B', x=150.0), NonnegativeElement('C')]
    reactants, products = {('A', 'x'): 1, ('B', 'x'): 1}, {('C', 'x'): 1}
    system = CompositeSystem(
        elements=species,
        interactions=[
            ReactionInteraction('binding', reactants=reactants, products=products, k=0.001, mode='langevin'),
        ],
    )
    trajectory = _run(system, 5.0, 0.05, solver=EulerMaruyamaSolver(dt=1e-3))
    difference = trajectory.values('A.x') - trajectory.values('B.x')
    np.testing.assert_allclose(difference, 50.0, rtol=1e-9)
    deterministic = _run(
        CompositeSystem(
            elements=species,
            interactions=[ReactionInteraction('binding', reactants=reactants, products=products, k=0.001)],
        ),
        5.0,
        0.05,
    )
    assert np.max(np.abs(trajectory.values('A.x') - deterministic.values('A.x'))) > 0.5


def test_reaction_of_unknown_mode_is_rejected():
    """Test that a reaction of an unknown mode raises an error that names it."""
    system = CompositeSystem(
        elements=[NonnegativeElement('A', x=10.0), NonnegativeElement('B', x=10.0), NonnegativeElement('C')],
        interactions=[
            ReactionInteraction(
                'binding',
                reactants={('A', 'x'): 1, ('B', 'x'): 1},
                products={('C', 'x'): 1},
                mode='stochastic',
            ),
        ],
    )
    with pytest.raises(ValidationError, match='binding'):
        PassCompiler().prepare(system).run()


def test_chemical_langevin_network_conserves_totals():
    """Test that the enzyme network under noise keeps `E + C` and `S + C + P` at every frame."""
    trajectory = simulate(ChemicalLangevinNetworkSystem(), random_state=0)
    enzyme = trajectory.values('E.x') + trajectory.values('C.x')
    substrate = trajectory.values('S.x') + trajectory.values('C.x') + trajectory.values('P.x')
    np.testing.assert_allclose(enzyme, 100.0, rtol=1e-9)
    np.testing.assert_allclose(substrate, 300.0, rtol=1e-9)
    assert np.std(np.diff(trajectory.values('C.x'))) > 0
