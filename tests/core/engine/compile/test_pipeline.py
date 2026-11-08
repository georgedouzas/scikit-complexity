"""Test the ordering of the passes of the general compiler by the fields they read and write, and their run."""

import pytest

from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumpyBackend
from skcomplex.core.engine.compile import BasePass
from skcomplex.core.engine.compile._compilation import Compilation
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.compile.passes._flatten import FlattenPass
from skcomplex.core.errors import PipelineError
from skcomplex.systems.biology.ecology import LotkaVolterraSystem


def _pass(name, reads, writes, log=None):
    """Return a pass that records its name when it runs."""

    def _get_reads(self):
        """Return the fields the pass reads."""
        return reads

    def _get_writes(self):
        """Return the fields the pass writes."""
        return writes

    def _run(self, inputs=None):
        """Record the run."""
        if log is not None:
            log.append(name)

    namespace = {'_get_reads': _get_reads, '_get_writes': _get_writes, '_run': _run, '__doc__': name}
    return type(name, (BasePass,), namespace)()


def _names(passes):
    """Return the class names of passes."""
    return [type(item).__name__ for item in passes]


def test_passes_are_ordered_produced_refined_then_read():
    """Test that a producer runs before the refiners of its field, and those before its readers."""
    passes = [
        _pass('Read', ('layout',), ('kernels',)),
        _pass('Refine', ('layout',), ('layout',)),
        _pass('Produce', ('nodes',), ('layout',)),
    ]
    assert _names(PassCompiler._ordered(passes, present=['nodes'])) == ['Produce', 'Refine', 'Read']


def test_passes_keep_the_given_order_of_refiners():
    """Test that refiners of one field run in the order they are given."""
    passes = [
        _pass('Produce', (), ('layout',)),
        _pass('SecondElement', ('layout',), ('layout',)),
        _pass('FirstElement', ('layout',), ('layout',)),
    ]
    assert _names(PassCompiler._ordered(passes)) == ['Produce', 'SecondElement', 'FirstElement']


def test_ordering_names_a_missing_input():
    """Test that a field no pass produces and the compilation lacks names the pass that reads it."""
    with pytest.raises(PipelineError, match="Lower reads the field 'layout'"):
        PassCompiler._ordered([_pass('Lower', ('layout',), ('lowered',))], present=['nodes'])


def test_ordering_accepts_a_field_the_compilation_holds():
    """Test that a field the compilation holds needs no producer."""
    assert _names(PassCompiler._ordered([_pass('Lower', ('layout',), ('lowered',))], present=['layout'])) == ['Lower']


def test_ordering_rejects_two_producers():
    """Test that two passes producing one field are rejected."""
    with pytest.raises(PipelineError, match='both produce'):
        PassCompiler._ordered([_pass('One', (), ('layout',)), _pass('Two', (), ('layout',))])


def test_ordering_rejects_a_cycle():
    """Test that passes reading each other's fields are rejected."""
    with pytest.raises(PipelineError, match='cycle'):
        PassCompiler._ordered([_pass('One', ('b',), ('a',)), _pass('Two', ('a',), ('b',))])


def test_passes_run_only_what_a_field_needs():
    """Test that running until a field runs the passes that produce it and their inputs, once each."""
    log = []
    passes = [
        _pass('Problem', ('lowered',), ('problem',), log),
        _pass('Lower', ('layout',), ('lowered',), log),
        _pass('Layout', ('system',), ('layout',), log),
        _pass('OtherElement', ('system',), ('census',), log),
    ]
    compiler = PassCompiler(passes=passes)
    compiler.passes_ = tuple(passes)
    compilation = compiler._new_compilation(System())
    compiler._run_passes(compilation, until='lowered')
    assert log == ['Layout', 'Lower']
    compiler._run_passes(compilation)
    assert log == ['Layout', 'Lower', 'Problem']


def test_pass_is_prepared_on_its_compilation_and_runs_on_it():
    """Test that a pass takes its compilation in `prepare` and changes it in `run`."""
    compilation = Compilation(LotkaVolterraSystem().prepare().system_)
    flatten = FlattenPass().prepare(compilation)
    assert flatten.compilation_ is compilation
    assert flatten.run() is flatten
    assert compilation.nodes


def test_compiler_flattens_the_system_once(monkeypatch):
    """Test that validating and compiling a system flattens its tree once."""
    calls = []
    flatten = FlattenPass._flatten_system

    def counted(*args, **kwargs):
        calls.append(args[0])
        return flatten(*args, **kwargs)

    monkeypatch.setattr(FlattenPass, '_flatten_system', staticmethod(counted))
    PassCompiler(backend=NumpyBackend()).prepare(LotkaVolterraSystem().prepare().system_).run()
    assert len(calls) == 1
