"""Test the `_satisfiability` module of `skcomplex.core.components.interactions`."""

import itertools
from typing import Annotated

import numpy as np
import pytest
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import SatisfiabilityInteraction
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import System
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import CardinalitySpec, ClauseSpec, StateSpec


class WeightElement(Element):
    """Member with a Boolean weight."""

    def __init__(self, label: str, on: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0) -> None:
        super().__init__(label)
        self.on = on


class RealWeightElement(Element):
    """Member with a real weight."""

    def __init__(self, label: str, on: Annotated[float, StateSpec()] = 0.0) -> None:
        super().__init__(label)
        self.on = on


def _pattern_constraint(pattern: np.ndarray, output: int) -> CardinalitySpec:
    """Return the constraint that at least a majority of the literals `w_i = output * pattern_i` hold."""
    literals = tuple(int(sign * index) for index, sign in enumerate(pattern * output, start=1))
    return CardinalitySpec(literals, (len(pattern) + 1) // 2)


def _perceptron(size: int, count: int, seed: int) -> tuple[System, np.ndarray, np.ndarray]:
    """Return a perceptron of Boolean weights that reveals random patterns, with the patterns and outputs."""
    rng = np.random.default_rng(seed)
    patterns = rng.choice([-1, 1], (count, size))
    outputs = rng.choice([-1, 1], count)
    constraints = [_pattern_constraint(pattern, int(output)) for pattern, output in zip(patterns, outputs, strict=True)]
    group = GroupElement('weights', element=WeightElement, size=size)
    decide = SatisfiabilityInteraction('decide', target='weights', state='on', constraints=constraints)
    return System(elements=[group], interactions=[decide], time=DiscreteTime()), patterns, outputs


def _holds(weights: np.ndarray, patterns: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """Return which patterns the signed weights classify as their outputs."""
    return np.sign(patterns @ weights) == outputs


def test_constraint_perceptron_pattern() -> None:
    """Test that a pattern of 21 weights becomes the constraint that at least 11 literals hold."""
    pattern = np.ones(21, dtype=int)
    pattern[3] = -1
    spec = _pattern_constraint(pattern, -1)
    assert spec.bound == 11
    assert spec.sense == 'at-least'
    assert spec.literals[0] == -1
    assert spec.literals[3] == 4


def test_prepare_compiler_real_state() -> None:
    """Test that a state that is not Boolean is reported with the interaction and the state."""
    group = GroupElement('weights', element=RealWeightElement, size=3)
    decide = SatisfiabilityInteraction('decide', target='weights', state='on', constraints=[ClauseSpec((1,))])
    system = System(elements=[group], interactions=[decide], time=DiscreteTime())
    found = [item for item in PassCompiler().prepare(system).report_.issues if item.code == 'law-domain']
    assert len(found) == 1
    assert 'decide' in found[0].message
    assert 'on' in found[0].message


def test_run_small_perceptron_matches_enumeration() -> None:
    """Test that the verdict of every step equals enumerating all weights and that assignments satisfy the patterns."""
    system, patterns, outputs = _perceptron(3, 8, 0)
    trajectory = simulate(system, until=8, random_state=0)
    verdicts = trajectory.values('decide.satisfiable')
    candidates = [np.asarray(item) for item in itertools.product([-1, 1], repeat=3)]
    for step in range(1, 9):
        revealed = patterns[:step], outputs[:step]
        expected = any(_holds(candidate, *revealed).all() for candidate in candidates)
        assert bool(verdicts[step]) == expected
        if expected:
            weights = np.asarray(trajectory.agents('weights', step).values['on'], dtype=float)
            assert _holds(2 * weights - 1, *revealed).all()


def test_run_capacity_of_binary_perceptron() -> None:
    """Test that the mean load at the first unsatisfiable step lies between 0.7 and 1.0 over 20 replicates."""
    loads = []
    for seed in range(20):
        system, _, _ = _perceptron(21, 40, seed)
        verdicts = simulate(system, until=40, random_state=seed).values(
            'decide.satisfiable',
        )
        loads.append(int(np.argmax(verdicts[1:] == 0)) + 1)
    assert 0.7 <= np.mean(loads) / 21 <= 1.0


def test_check_refuses_constraint_that_is_no_spec() -> None:
    """Test that a constraint that is not a spec is refused."""
    with pytest.raises(ValidationError, match='ClauseSpec'):
        SatisfiabilityInteraction('decide', target='weights', state='on', constraints=[(1, 2)]).prepare()
