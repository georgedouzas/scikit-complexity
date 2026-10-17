"""Run the Python examples of the README and the user guide."""

import re
from pathlib import Path

import pytest
import rerun as rr

ROOT = Path(__file__).parent.parent
DOCUMENTS = [ROOT / 'README.md', ROOT / 'docs' / 'overview' / 'user_guide.md']
CODE_BLOCK = re.compile(r'```python\n(.*?)```', re.DOTALL)


@pytest.mark.parametrize('document', DOCUMENTS, ids=lambda path: path.name)
def test_documentation_examples_run(document, monkeypatch):
    """Test that the Python blocks of a document run in order without a viewer."""
    monkeypatch.setattr(rr.RecordingStream, 'spawn', lambda *_, **__: None)
    namespace: dict[str, object] = {}
    for block in CODE_BLOCK.findall(document.read_text()):
        code = compile(block, str(document), 'exec')
        exec(code, namespace)  # noqa: S102  # nosec B102: runs the repository's own documentation
