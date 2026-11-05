"""Test the `_source_loader` module of `skcomplex.core.engine.problems`."""

import pytest

from skcomplex.core.engine.problems import SourceLoader
from skcomplex.core.engine.problems._source_loader import CACHE_DIR
from skcomplex.core.errors import ValidationError


def test_source_loader_imports_a_function(tmp_path):
    """Test that a loader imports the function of generated source from a file under its directory."""
    loader = SourceLoader('skcomplex_test').prepare(tmp_path)
    assert loader.run('def triple(x):\n    return 3 * x\n').module_.triple(2) == 6
    assert len(list(tmp_path.glob('skcomplex_test_*.py'))) == 1


def test_source_loader_defaults_to_the_cache_directory():
    """Test that a loader prepared with no directory writes to the default cache directory."""
    assert SourceLoader('skcomplex_test').prepare().directory_ == CACHE_DIR


def test_source_loader_injects_functions(tmp_path):
    """Test that injected functions are read by name from the source."""
    loader = SourceLoader('skcomplex_test', {'offset': lambda x: x + 1}).prepare(tmp_path)
    assert loader.run('def shifted(x):\n    return offset(x) * 2\n').module_.shifted(1) == 4


def test_source_loader_rejects_a_prefix_that_is_not_an_identifier():
    """Test that a prefix with a dash is rejected."""
    with pytest.raises(ValidationError, match='identifier'):
        SourceLoader('not-valid').prepare()
