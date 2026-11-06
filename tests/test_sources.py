"""Test the sources of the repository: the package, its tests, and its documentation hide no finding of the checks."""

import ast
import multiprocessing
import re
from pathlib import Path

import pytest

FOLDERS = ('src', 'tests', 'docs')
README = 'README.md'
USER_GUIDE = Path('docs') / 'overview' / 'user_guide'
PYTHON_BLOCK = re.compile(r'^```python[^\n]*\n(.*?)^```$', re.MULTILINE | re.DOTALL)
SYSTEMS = 'skcomplex.systems'
CATALOG_FUNCTIONS = ('catalog', 'describe', 'create_system', 'create_simulation')
FIELD_MODULE = re.compile(rf'{re.escape(SYSTEMS)}\.[\w<]')
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


def _documents(root: Path) -> list[Path]:
    """Return the README and the pages of the user guide, relative to the root."""
    return [Path(README), *sorted(path.relative_to(root) for path in (root / USER_GUIDE).rglob('*.md'))]


def _python_blocks(path: Path) -> list[tuple[int, str]]:
    """Return the Python blocks of a Markdown file, each with the line of the file before its first line."""
    text = path.read_text(encoding='utf-8')
    return [(text.count('\n', 0, match.start(1)), match.group(1)) for match in PYTHON_BLOCK.finditer(text)]


def _imported_names(code: str) -> list[tuple[int, str, str]]:
    """Return what a block of code imports, each as its line in the block, its module, and its name."""
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.ImportFrom) and node.module:
            found += [(node.lineno, node.module, alias.name) for alias in node.names]
        elif isinstance(node, ast.Import):
            found += [(node.lineno, alias.name, '') for alias in node.names]
    return found


def _is_field_import(module: str, name: str) -> bool:
    """Return whether an import reaches into the catalog, beyond the functions of the catalog."""
    if module.startswith(f'{SYSTEMS}.'):
        return True
    if module == SYSTEMS:
        return name not in CATALOG_FUNCTIONS
    return module == 'skcomplex' and name == 'systems'


def _field_imports(root: Path) -> list[str]:
    """Return every import of the Python blocks of the documents beyond the catalog, as `file:line: module name`."""
    return [
        f'{document}:{start + line}: {module} {name}'
        for document in _documents(root)
        for start, code in _python_blocks(root / document)
        for line, module, name in _imported_names(code)
        if _is_field_import(module, name)
    ]


def _field_mentions(root: Path) -> list[str]:
    """Return every line of the documents that names a field module, as `file:line: text`."""
    return [
        f'{document}:{number}: {line.strip()}'
        for document in _documents(root)
        for number, line in enumerate((root / document).read_text(encoding='utf-8').splitlines(), start=1)
        if FIELD_MODULE.search(line)
    ]


def test_documents_import_from_no_field_module(request):
    """Test that the Python blocks of the README and the user guide import only the catalog functions of the systems."""
    found = _field_imports(request.config.rootpath)
    assert not found, f'{len(found)} imports beyond the catalog functions:\n' + '\n'.join(found)


def test_documents_name_no_field_module(request):
    """Test that no line of the README and the user guide names a field module, in code or in prose."""
    found = _field_mentions(request.config.rootpath)
    assert not found, f'{len(found)} lines name a field module:\n' + '\n'.join(found)


def test_readme_custom_system_runs_from_the_core_alone(request):
    """Test that the first custom system of the README simulates in a fresh interpreter without the catalog."""
    text = (request.config.rootpath / README).read_text(encoding='utf-8')
    section = text[text.index('### Build a custom system') :]
    code = PYTHON_BLOCK.search(section).group(1)
    assert 'Simulation' in code
    check = f"""
import sys
{code}
loaded = [name for name in sys.modules if name == {SYSTEMS!r} or name.startswith({SYSTEMS + '.'!r})]
assert loaded == [], loaded
"""
    process = multiprocessing.get_context('spawn').Process(target=exec, args=(check, {'__name__': '__main__'}))
    process.start()
    process.join()
    assert process.exitcode == 0
