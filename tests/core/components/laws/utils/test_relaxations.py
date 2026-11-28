"""Test the `_relaxations` module of `skcomplex.core.components.laws.utils`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import relax
from skcomplex.core.components.laws.utils._relaxations import relaxation_of
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace, RadiusNeighborhood
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System

CROSS = RadiusNeighborhood(1, metric='manhattan', center=True)
LINE = LatticeNeighborhood(((-1,), (0,), (1,)))
VALUES = {'neurons': 'potential', 'pile': 'height'}


def _avalanche(s):
    """Return the potentials driven at the first neuron and relaxed until no neuron fires."""
    potential = Interaction._get_symbolic(s.target, s.t).potential
    fire = UpdateLaw(
        potential,
        lambda left, value, right, **kwargs: value * (value < 1) + 0.5 * (left >= 1) + 0.5 * (right >= 1),
        neighborhood=LINE,
    )
    drive = UpdateLaw(potential, lambda values, **kwargs: values + 0.6 * (np.arange(values.size) == 0), arrays=True)
    relaxation = relax(potential, fire, lambda values: bool((values < 1).all()), drive=drive)
    return [UpdateLaw(potential, relaxation, label='avalanche')]


def _run(label, space, laws, until):
    """Return the run of one value at every site of a space under the laws of one interaction."""
    rule = Interaction('laws', roles={'target': label}, laws=laws)
    system = System(elements=[Element(label, values={VALUES[label]: 0.0}, space=space)], interactions=[rule])
    return system.prepare().simulate(until=until, dt=1)


def test_relax_keeps_its_rules_drive_and_settings():
    """Test that a relaxation keeps its rules and drive in order, its stability, and its largest iterations."""
    height = var('height')
    topple = UpdateLaw(height, lambda z, *n, **kwargs: z, neighborhood=CROSS)
    settle = UpdateLaw(height, lambda z, *n, **kwargs: z, neighborhood=CROSS)
    drop = UpdateLaw(height, lambda values, **kwargs: values + 1, arrays=True)
    relaxation = relaxation_of(relax(height, [topple, settle], bool, drive=drop, max_iterations=7))
    assert relaxation.rules == (topple, settle)
    assert relaxation.drive == (drop,)
    assert relaxation.max_iterations == 7
    assert relaxation_of(height + 1) is None


def test_relax_needs_laws():
    """Test that a relaxation needs a rule, and takes laws only."""
    height = var('height')
    with pytest.raises(ValidationError, match='no rules'):
        relax(height, [], bool)
    with pytest.raises(ValidationError, match='laws'):
        relax(height, [lambda z: z], bool)


def test_relax_records_a_neuronal_avalanche():
    """Test that a driven neuron fires an avalanche, recorded at the start of its step with its size."""
    trajectory = _run('neurons', LatticeSpace(4, boundary='fixed'), _avalanche, 2)
    (record,) = trajectory.events
    assert (record.label, record.time) == ('avalanche', 1.0)
    assert record.size >= 1
    assert np.all(trajectory.states[-1] < 1)


def test_relax_checks_every_law_it_holds():
    """Test that a run checks every rule of a relaxation."""

    def laws(s):
        """Return the relaxation with the unknown schedule."""
        height = Interaction._get_symbolic(s.target, s.t).height
        settle = UpdateLaw(height, lambda z, *n, **kwargs: z, neighborhood=CROSS, schedule='parallel')
        return [UpdateLaw(height, relax(height, settle, bool))]

    with pytest.raises(ValidationError, match='Unknown schedule'):
        _run('pile', LatticeSpace((2, 2)), laws, 1)


def _sandpile(symbolic):
    """Return the laws of a sandpile that gains a grain at every site, by an update law or by an array update."""

    def laws(s):
        height = s.target.height
        topple = UpdateLaw(
            height,
            lambda z, *n, **kwargs: z - 4 * (z >= 4) + sum(m >= 4 for m in n[1:]),
            neighborhood=CROSS,
        )
        drive = UpdateLaw(height, height + 1) if symbolic else UpdateLaw(height, lambda z, **kwargs: z + 1, arrays=True)
        return [UpdateLaw(height, relax(height, topple, lambda z: bool((z < 4).all()), drive=drive))]

    return laws


def _is_stable(heights):
    """Return whether no site holds four grains."""
    return bool((heights < 4).all())


def test_relax_records_the_profile_of_a_single_toppling():
    """Test that one site of four grains topples once, with the size 1, the duration 1, and the profile `(1,)`."""

    def laws(s):
        """Return the Bak-Tang-Wiesenfeld rule relaxed until no site holds four grains."""
        height = Interaction._get_symbolic(s.target, s.t).height
        topple = UpdateLaw(
            height,
            lambda z, *n, **kwargs: z - 4 * (z >= 4) + sum(m >= 4 for m in n[1:]),
            neighborhood=CROSS,
        )
        return [UpdateLaw(height, relax(height, topple, _is_stable, activity=lambda z: z >= 4), label='avalanche')]

    heights = np.zeros(9)
    heights[4] = 4
    rule = Interaction('laws', roles={'target': 'pile'}, laws=laws)
    space = LatticeSpace((3, 3), boundary='fixed')
    system = System(elements=[Element('pile', values={'height': heights}, space=space)], interactions=[rule])
    (record,) = system.prepare().simulate(until=1, dt=1).events
    assert (record.size, record.duration, record.profile) == (1, 1, (1,))


def test_relax_reads_the_values_of_the_namespace_and_an_update_law_drive():
    """Test that a relaxation of `s.target.height` driven by `UpdateLaw(height, height + 1)` runs as an array drive."""
    space = LatticeSpace((4, 4), boundary='fixed')
    symbolic = _run('pile', space, _sandpile(symbolic=True), 6).get_values('pile.height')
    arrays = _run('pile', space, _sandpile(symbolic=False), 6).get_values('pile.height')
    np.testing.assert_array_equal(symbolic, arrays)
    assert np.all(symbolic[-1] < 4)
    assert symbolic[-1].sum() < 6 * 16
