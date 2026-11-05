"""Test the sources of the repository: the package, its tests, and its documentation hide no finding of the checks."""

import ast
import multiprocessing
import re
from pathlib import Path

import pytest

FOLDERS = ('src', 'tests', 'docs')
DOCUMENTS = ('README.md', 'docs/overview/user_guide.md')
PYTHON_BLOCK = re.compile(r'^```python[^\n]*\n(.*?)^```$', re.MULTILINE | re.DOTALL)
MODELS = 'skcomplex.models'
GENERATED = Path('docs') / 'generated'
FORBIDDEN = (
    'TYPE' + '_CHECKING',
    'no' + 'qa',
    'type' + ': ignore',
    'no' + 'sec',
)


def _sources(root: Path) -> list[Path]:
    """Return the Python files of the package, its tests, and its documentation, without the generated pages."""
    return [
        path
        for folder in FOLDERS
        for path in sorted((root / folder).rglob('*.py'))
        if not path.relative_to(root).is_relative_to(GENERATED)
    ]


@pytest.mark.parametrize('word', FORBIDDEN)
def test_no_source_hides_a_finding(request, word):
    """Test that no source imports for type checking only or suppresses a finding of a linter or a type checker."""
    root = request.config.rootpath
    found = [
        f'{path.relative_to(root)}:{number}'
        for path in _sources(root)
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1)
        if word in line
    ]
    assert not found, f'{word} appears in: {", ".join(found)}'


def _python_blocks(path: Path) -> list[tuple[int, str]]:
    """Return the Python blocks of a Markdown file, each with the line of the file before its first line."""
    text = path.read_text(encoding='utf-8')
    return [(text.count('\n', 0, match.start(1)), match.group(1)) for match in PYTHON_BLOCK.finditer(text)]


def _imported_modules(code: str) -> list[tuple[int, str]]:
    """Return the modules a block of code imports, each with its line in the block."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
    return found


def _field_imports(root: Path) -> list[str]:
    """Return every import from a field module in the Python blocks of the documents, as `file:line: module`."""
    return [
        f'{document}:{start + line}: {module}'
        for document in DOCUMENTS
        for start, code in _python_blocks(root / document)
        for line, module in _imported_modules(code)
        if module.startswith(f'{MODELS}.')
    ]


def test_documents_import_from_no_field_module(request):
    """Test that the Python blocks of the README and the user guide import nothing from a field module."""
    found = _field_imports(request.config.rootpath)
    assert not found, f'{len(found)} imports from field modules:\n' + '\n'.join(found)


def test_readme_custom_model_runs_from_the_core_alone(request):
    """Test that the first custom model of the README simulates in a fresh interpreter without the model library."""
    text = (request.config.rootpath / 'README.md').read_text(encoding='utf-8')
    section = text[text.index('### Build a custom model') :]
    code = PYTHON_BLOCK.search(section).group(1)
    assert 'Simulation' in code
    check = f"""
import sys
{code}
loaded = [name for name in sys.modules if name == {MODELS!r} or name.startswith({MODELS + '.'!r})]
assert loaded == [], loaded
"""
    process = multiprocessing.get_context('spawn').Process(target=exec, args=(check, {'__name__': '__main__'}))
    process.start()
    process.join()
    assert process.exitcode == 0
