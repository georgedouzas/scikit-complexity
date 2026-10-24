"""Test the ordering of the passes of a compilation by the fields they read and write, and their run."""

import sys

import pytest

from skcomplex.core.base import PipelineError
from skcomplex.core.system import System
from skcomplex.engine.compile import Compilation, Pass, compile_system, order_passes, run_passes
from skcomplex.models import create


def _pass(name, reads, writes, log=None):
    """Return a pass that records its name when it runs."""

    def run(self, compilation):
        """Record the run."""
        if log is not None:
            log.append(name)

    return type(name, (Pass,), {'reads': reads, 'writes': writes, 'run': run, '__doc__': name})()


def _names(passes):
    """Return the class names of passes."""
    return [type(item).__name__ for item in passes]


def test_order_passes_produces_refines_then_reads():
    """Test that a field is produced, then refined, then read, whatever the order the passes are given in."""
    passes = [
        _pass('Read', ('layout',), ('kernels',)),
        _pass('Refine', ('layout',), ('layout',)),
        _pass('Produce', ('nodes',), ('layout',)),
    ]
    assert _names(order_passes(passes, present=['nodes'])) == ['Produce', 'Refine', 'Read']


def test_order_passes_keeps_the_given_order_of_refiners():
    """Test that refiners of one field that the fields leave unordered run in the order they are given."""
    passes = [
        _pass('Produce', (), ('lowered',)),
        _pass('Second', ('lowered',), ('lowered',)),
        _pass('First', ('lowered',), ('lowered',)),
    ]
    assert _names(order_passes(passes)) == ['Produce', 'Second', 'First']


def test_order_passes_names_a_missing_input():
    """Test that a pass reading a field that no pass produces and the compilation does not hold is named."""
    with pytest.raises(PipelineError, match="The pass Lower reads the field 'layout', which no pass produces"):
        order_passes([_pass('Lower', ('layout',), ('lowered',))], present=['nodes'])


def test_order_passes_accepts_a_field_the_compilation_holds():
    """Test that a field the compilation already holds needs no producer."""
    assert _names(order_passes([_pass('Lower', ('layout',), ('lowered',))], present=['layout'])) == ['Lower']


def test_order_passes_rejects_two_producers():
    """Test that two passes producing one field are named."""
    with pytest.raises(PipelineError, match='both produce the field'):
        order_passes([_pass('One', (), ('layout',)), _pass('Two', (), ('layout',))])


def test_order_passes_rejects_a_cycle():
    """Test that passes reading each other's fields are named as a cycle."""
    with pytest.raises(PipelineError, match='in a cycle'):
        order_passes([_pass('One', ('b',), ('a',)), _pass('Two', ('a',), ('b',))])


def test_run_passes_runs_only_what_a_field_needs():
    """Test that running until a field runs the passes that produce it and their inputs, once each."""
    log = []
    passes = [
        _pass('Problem', ('lowered',), ('problem',), log),
        _pass('Lower', ('layout',), ('lowered',), log),
        _pass('Layout', ('system',), ('layout',), log),
        _pass('Other', ('system',), ('census',), log),
    ]
    compilation = Compilation(System())
    run_passes(compilation, passes, until='lowered')
    assert log == ['Layout', 'Lower']
    run_passes(compilation, passes)
    assert log == ['Layout', 'Lower', 'Problem']


def test_compile_system_flattens_the_system_once(monkeypatch):
    """Test that validating and compiling a system flattens its tree once."""
    calls = []
    for name, module in list(sys.modules.items()):
        flatten = getattr(module, 'flatten_system', None)
        if name.startswith('skcomplex.core') and callable(flatten):

            def counted(*args, flatten=flatten, **kwargs):
                calls.append(args[0])
                return flatten(*args, **kwargs)

            monkeypatch.setattr(module, 'flatten_system', counted)
    compile_system(create('lotka_volterra'), backend='numpy')
    assert len(calls) == 1
