"""Run the Python examples of the README and the user guide."""

import re
import sys
import types
from pathlib import Path

import pytest
import rerun as rr

ROOT = Path(__file__).parent.parent
DOCUMENTS = [ROOT / 'README.md', ROOT / 'docs' / 'overview' / 'user_guide.md']
CODE_BLOCK = re.compile(r'```python\n(.*?)```', re.DOTALL)


@pytest.mark.parametrize('document', DOCUMENTS, ids=lambda path: path.name)
def test_documentation_examples_run(document, monkeypatch):
    """Test that the Python blocks of a document run in order, as one module, without a viewer."""
    monkeypatch.setattr(rr, 'spawn', lambda **_: None)
    module = types.ModuleType(f'documentation_{document.stem.lower()}')
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for block in CODE_BLOCK.findall(document.read_text()):
        code = compile(block, str(document), 'exec')
        exec(code, module.__dict__)  # noqa: S102  # nosec B102: runs the repository's own documentation
