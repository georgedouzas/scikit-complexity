"""Test the codemod that rewrites imports from tables of old and new import paths."""

import importlib.util
import json
from pathlib import Path

import pytest

TOOL = Path(__file__).parents[2] / 'tools' / 'rewrite_imports.py'
TOPOLOGY = """
[[moves]]
old = "pkg.core.laws"
new = "pkg.core.formalisms.topology"
names = ["TopologyRequest", "add_edge"]
"""
RENEWALS = {'moves': [{'old': 'pkg.core.compile._renewals', 'new': 'pkg.core.formalisms.renewals._passes'}]}
RELATIVE = """\"\"\"Planted module.\"\"\"

from ..formalisms.topology import rewire
from ..laws import Rate, TopologyRequest, add_edge  # noqa: F401
from ._renewals import CompiledRenewal


def f():
    \"\"\"Return the names.

    Examples:
        >>> from pkg.core.laws import Rate, add_edge
        >>> add_edge
    \"\"\"
    from ..laws import add_edge as edge

    return Rate, TopologyRequest, add_edge, edge, rewire, CompiledRenewal
"""
ABSOLUTE = """from pkg.core.compile._renewals.clock import Clock
from pkg.core.laws import (
    Noise,
    Rate,
    TopologyRequest,
)

print(Clock, Noise, Rate, TopologyRequest)
"""
MARKDOWN = """# Guide

```python
from pkg.core.laws import Rate, add_edge
```

```pycon
>>> from pkg.core.laws import add_edge
```

    from pkg.core.laws import add_edge
"""


def _tool():
    """Return the codemod module loaded from its file."""
    spec = importlib.util.spec_from_file_location('rewrite_imports', TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def planted(tmp_path: Path) -> Path:
    """Return a directory with a planted package, a script, a Markdown page, and two tables."""
    package = tmp_path / 'pkg' / 'core' / 'compile'
    package.mkdir(parents=True)
    for directory in (tmp_path / 'pkg', tmp_path / 'pkg' / 'core', package):
        (directory / '__init__.py').write_text('')
    (package / '_planted.py').write_text(RELATIVE)
    (tmp_path / 'script.py').write_text(ABSOLUTE)
    (tmp_path / 'guide.md').write_text(MARKDOWN)
    tables = tmp_path / 'moves'
    tables.mkdir()
    (tables / 'topology.toml').write_text(TOPOLOGY)
    (tables / 'renewals.json').write_text(json.dumps(RENEWALS))
    (tmp_path / 'pyproject.toml').write_text('[tool.black]\nline-length = 120\nskip-string-normalization = true\n')
    return tmp_path


def test_main_splits_merges_and_keeps_relative_imports(planted):
    """Test that a relative import splits by destination, merges with one of the same module, and stays relative."""
    _tool().main(['--table', str(planted / 'moves'), '--pyproject', str(planted / 'pyproject.toml'), str(planted)])
    text = (planted / 'pkg' / 'core' / 'compile' / '_planted.py').read_text()
    assert 'from ..formalisms.topology import rewire, TopologyRequest, add_edge\n' in text
    assert 'from ..laws import Rate  # noqa: F401\n' in text
    assert 'from ..formalisms.renewals._passes import CompiledRenewal\n' in text
    assert '    from ..formalisms.topology import add_edge as edge\n' in text
    assert (
        '        >>> from pkg.core.laws import Rate\n        >>> from pkg.core.formalisms.topology import add_edge\n'
        in (text)
    )


def test_main_rewrites_absolute_imports_and_module_moves(planted):
    """Test that an absolute import splits, a submodule of a moved module moves, and black formats the result."""
    _tool().main(['--table', str(planted / 'moves'), '--pyproject', str(planted / 'pyproject.toml'), str(planted)])
    text = (planted / 'script.py').read_text()
    assert text.startswith(
        'from pkg.core.formalisms.renewals._passes.clock import Clock\n'
        'from pkg.core.laws import Noise, Rate\n'
        'from pkg.core.formalisms.topology import TopologyRequest\n',
    )


def test_main_rewrites_markdown_code_blocks(planted):
    """Test that Python and console blocks of Markdown are rewritten, and indented text outside blocks is not."""
    _tool().main(['--table', str(planted / 'moves'), '--pyproject', str(planted / 'pyproject.toml'), str(planted)])
    text = (planted / 'guide.md').read_text()
    assert '```python\nfrom pkg.core.laws import Rate\nfrom pkg.core.formalisms.topology import add_edge\n```' in text
    assert '```pycon\n>>> from pkg.core.formalisms.topology import add_edge\n```' in text
    assert text.endswith('    from pkg.core.laws import add_edge\n')


def test_rewrite_source_wraps_long_doctest_imports():
    """Test that a doctest import too long for one line becomes a parenthesized import with continuation lines."""
    tool = _tool()
    names = [f'name_{index:03d}' for index in range(20)]
    moves = [tool.Move('a', 'b', frozenset(names))]
    text = tool.rewrite_source(f'"""Doc.\n\n>>> from a import {', '.join(names)}\n"""\n', moves)
    assert '>>> from b import (\n...     name_000,\n' in text
    assert '...     name_019,\n... )\n"""\n' in text


def test_rewrite_source_leaves_unmoved_imports():
    """Test that source without moved names is returned unchanged."""
    tool = _tool()
    source = 'from a import x  # keep\n'
    assert tool.rewrite_source(source, [tool.Move('a', 'b', frozenset({'y'}))]) == source


def test_load_moves_rejects_incomplete_moves(tmp_path):
    """Test that a move without its new module is rejected and named."""
    table = tmp_path / 'broken.toml'
    table.write_text('[[moves]]\nold = "a"\n')
    with pytest.raises(ValueError, match='misses its old or its new module'):
        _tool().load_moves([table])
