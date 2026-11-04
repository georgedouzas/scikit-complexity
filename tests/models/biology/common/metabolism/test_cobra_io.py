"""Test the `_cobra_io` module of `skcomplex.models.biology.common.metabolism`."""

import multiprocessing

import pytest

from skcomplex.core.spaces import StaticTime
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.models.biology.common.metabolism import (
    from_cobra,
    load_network,
    save_network,
)

AEROBIC = 0.8739215
ANAEROBIC = 0.2116629
HIDE_EXTRA = """
import sys

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.partition('.')[0] in {'cobra', 'straindesign', 'micom', 'optlang'}:
            raise ImportError(f'{name} is hidden')
        return None

sys.meta_path.insert(0, Blocker())
"""


def _core_system(**values):
    """Return the system of the E. coli core model."""
    return System(elements=[load_network('textbook', 'cell').set_params(**values)], time=StaticTime())


def _objective(system):
    """Return the objective of the only program of a static system."""
    trajectory = Simulation(system).check().prepare().run(0.0).trajectory_
    return float(trajectory.values('cell.objective_0')[0])


def test_bundled_core_model_grows_with_and_without_oxygen():
    """Test the aerobic and anaerobic growth of the bundled E. coli core model in the core solve."""
    assert _objective(_core_system()) == pytest.approx(AEROBIC, abs=1e-4)
    assert _objective(_core_system(lower_EX_o2_e=0.0)) == pytest.approx(ANAEROBIC, abs=1e-4)


@pytest.mark.parametrize('suffix', ['xml', 'json'])
def test_round_trip_keeps_the_network(tmp_path, suffix):
    """Test that a network written to a file and read back has the same matrix, bounds, rules, and optimum."""
    original = load_network('textbook', 'cell')
    path = tmp_path / f'network.{suffix}'
    save_network(original, path)
    copy = load_network(path, 'cell')
    assert set(copy.reactions) == set(original.reactions)
    for name, spec in original.reactions.items():
        other = copy.reactions[name]
        assert (other.lower, other.upper) == (spec.lower, spec.upper)
        assert set(other.rule.replace('(', '').replace(')', '').split()) == set(
            spec.rule.replace('(', '').replace(')', '').split(),
        )
        for metabolite, value in spec.coefficients.items():
            assert other.coefficients[metabolite] == pytest.approx(value)
    optimum = _objective(System(elements=[copy], time=StaticTime()))
    assert optimum == pytest.approx(AEROBIC, abs=1e-6)


def test_save_network_rejects_an_unknown_suffix(tmp_path):
    """Test that a path whose suffix names no format raises an error."""
    with pytest.raises(ValidationError, match=r'\.csv'):
        save_network(load_network('textbook', 'cell'), tmp_path / 'network.csv')


def test_from_cobra_rejects_another_object():
    """Test that an object that is not a COBRApy model raises an error."""
    with pytest.raises(ValidationError, match=r'cobra\.Model'):
        from_cobra(object())


@pytest.mark.network
def test_load_network_downloads_a_bigg_model():
    """Test that a BiGG identifier downloads a model of Mycobacterium tuberculosis."""
    assert len(load_network('iNJ661', 'cell').reactions) == 1025


def test_default_install_stays_importable():
    """Test that the catalog imports without the extra, lists its models, and the toy network runs."""
    code = HIDE_EXTRA + (
        'from skcomplex.engine.runtime import Simulation\n'
        'from skcomplex.models import catalog, create\n'
        'names = {item.name for item in catalog()}\n'
        "assert {'dynamic_fba', 'comets', 'community_fba'} <= names\n"
        "system = create('flux_balance')\n"
        "trajectory = Simulation(system).check().prepare().run(0.0).trajectory_\n"
        "assert abs(float(trajectory.values('cell.v_vbio')[0]) - 50) < 1e-9\n"
        "assert not {'cobra', 'straindesign', 'micom'} & set(sys.modules)\n"
        'from skcomplex.errors import MissingExtraError\n'
        'try:\n'
        "    create('dynamic_fba')\n"
        'except MissingExtraError as error:\n'
        "    assert 'dynamic_fba' in str(error) and 'metabolism' in str(error)\n"
        'else:\n'
        '    raise AssertionError\n'
    )
    process = multiprocessing.get_context('spawn').Process(target=exec, args=(code,))
    process.start()
    process.join()
    assert process.exitcode == 0


def test_network_loads_after_gap_has_run():
    """Test that reading a network in a process that has already run GAP does not crash it."""
    code = (
        'from skcomplex.core.spaces import CayleySpace\n'
        'from skcomplex.models.biology.common.metabolism import load_network\n'
        'from skcomplex.models.computer_science.automata import free_group\n'
        'CayleySpace(free_group(2), ball=2).check().prepare()\n'
        "assert load_network('textbook') is not None\n"
    )
    process = multiprocessing.get_context('spawn').Process(target=exec, args=(code,))
    process.start()
    process.join()
    assert process.exitcode == 0
