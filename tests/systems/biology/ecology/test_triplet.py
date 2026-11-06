"""Test the `_triplet` module of `skcomplex.systems.biology.ecology`."""

from itertools import combinations

import numpy as np

from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumpyBackend
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.systems.biology.common.populations import PopulationElement
from skcomplex.systems.biology.ecology import TripletInteraction


def test_triplet_interaction_sums_over_the_pairs_of_other_species():
    """Test that the rate of a species is `x_i sum_(j<k) a_ijk x_j x_k` over the pairs of other species."""
    rng = np.random.default_rng(0)
    size = 6
    tensor = np.zeros((size, size, size))
    for i in range(size):
        for j, k in combinations(set(range(size)) - {i}, 2):
            tensor[i, j, k] = tensor[i, k, j] = rng.normal()
    x = rng.random(size)
    species = GroupElement('species', element=PopulationElement, size=size, values={'n': x})
    triplets = TripletInteraction('triplets', target='species', a=tensor, state='n')
    problem = (
        PassCompiler(backend=NumpyBackend()).prepare(System(elements=[species], interactions=[triplets])).run().problem_
    )
    rate = problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 'species.n')
    expected = [
        x[i] * sum(tensor[i, j, k] * x[j] * x[k] for j, k in combinations(set(range(size)) - {i}, 2))
        for i in range(size)
    ]
    np.testing.assert_allclose(rate, expected, atol=1e-12)
