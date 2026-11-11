"""Test the `_flatten` module of `skcomplex.core.engine.compile.passes`."""

from skcomplex.core.engine.compile._compilation import Compilation
from skcomplex.core.engine.compile.passes._flatten import FlattenPass
from skcomplex.systems.biology.ecology import LotkaVolterraSystem


def test_pass_is_prepared_on_its_compilation_and_runs_on_it():
    """Test that a pass takes its compilation in `prepare` and changes it in `run`."""
    compilation = Compilation(LotkaVolterraSystem().prepare().system_)
    flatten = FlattenPass().prepare(compilation)
    assert flatten.compilation_ is compilation
    assert flatten.run() is flatten
    assert compilation.nodes
