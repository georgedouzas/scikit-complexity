"""Test the sources of the repository: the package, its tests, and its documentation hide no finding of the checks."""

import ast
import importlib
import multiprocessing
import re
from pathlib import Path
from typing import cast, is_typeddict

import pytest

from skcomplex.core.components.interactions import BaseInteraction
from skcomplex.core.components.laws import RuleLaw

FOLDERS = ('src', 'tests', 'docs')
README = 'README.md'
USER_GUIDE = Path('docs') / 'overview' / 'user_guide'
PYTHON_BLOCK = re.compile(r'^```python[^\n]*\n(.*?)^```$', re.MULTILINE | re.DOTALL)
SYSTEMS = 'skcomplex.systems'
CATALOG_FUNCTIONS = ('all_systems',)
GENERAL = f'{SYSTEMS}.general'
SUBFIELD_PACKAGE = re.compile(rf'{re.escape(SYSTEMS)}\.[a-z]\w*\.[a-z]\w*')
SYSTEMS_PATH = re.compile(rf'{re.escape(SYSTEMS)}\.[\w<>.]*[\w>]')
SUBFIELD_PLACEHOLDER = f'{SYSTEMS}.<field>.<subfield>'
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


def _is_allowed_import(module: str, name: str) -> bool:
    """Return whether an import takes a catalog function, a system class from its subfield, or a general realization."""
    if module == SYSTEMS:
        return name in CATALOG_FUNCTIONS
    if module.startswith(f'{GENERAL}.'):
        return SUBFIELD_PACKAGE.fullmatch(module) is not None
    if module.startswith(f'{SYSTEMS}.'):
        return SUBFIELD_PACKAGE.fullmatch(module) is not None and name.endswith('System')
    return not (module == 'skcomplex' and name == 'systems')


def _disallowed_imports(root: Path) -> list[str]:
    """Return every import of the Python blocks of the documents that the rule forbids, as `file:line: module name`."""
    return [
        f'{document}:{start + line}: {module} {name}'
        for document in _documents(root)
        for start, code in _python_blocks(root / document)
        for line, module, name in _imported_names(code)
        if not _is_allowed_import(module, name)
    ]


def _is_allowed_path(path: str) -> bool:
    """Return whether a dotted path below the systems names a subfield package, the one place to import a system."""
    return path == SUBFIELD_PLACEHOLDER or SUBFIELD_PACKAGE.fullmatch(path) is not None


def _disallowed_paths(root: Path) -> list[str]:
    """Return every line of the documents that names a path below the systems other than a subfield package."""
    return [
        f'{document}:{number}: {line.strip()}'
        for document in _documents(root)
        for number, line in enumerate((root / document).read_text(encoding='utf-8').splitlines(), start=1)
        if not all(_is_allowed_path(path) for path in SYSTEMS_PATH.findall(line))
    ]


def test_documents_import_systems_from_their_subfield(request):
    """Test that the README and the user guide import system classes from their subfield and nothing else of a field."""
    found = _disallowed_imports(request.config.rootpath)
    assert not found, f'{len(found)} imports beyond the system classes of a subfield:\n' + '\n'.join(found)


def test_documents_name_only_subfield_packages(request):
    """Test that the README and the user guide name no path below the systems other than a subfield package."""
    found = _disallowed_paths(request.config.rootpath)
    assert not found, f'{len(found)} lines name a path below the systems other than a subfield:\n' + '\n'.join(found)


def test_readme_custom_system_runs_from_the_core_alone(request):
    """Test that the first custom system of the README simulates in a fresh interpreter without the catalog."""
    text = (request.config.rootpath / README).read_text(encoding='utf-8')
    section = text[text.index('### Build a custom system') :]
    code = PYTHON_BLOCK.search(section).group(1)
    assert 'simulate(' in code
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


def _imported_modules(path: Path, package: str) -> list[str]:
    """Return the absolute names of the modules that a source file of a package imports, read from its syntax tree."""
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package.split('.')[: len(package.split('.')) - node.level + 1] if node.level else []
            module = '.'.join([*base, *([node.module] if node.module else [])])
            found += [module] + [f'{module}.{alias.name}' for alias in node.names]
    return found


def test_core_never_imports_the_systems(request):
    """Test that no module under `src/skcomplex/core` imports `skcomplex.systems`."""
    source = request.config.rootpath / 'src'
    found = [
        f'{path.relative_to(source)}: {module}'
        for path in sorted((source / 'skcomplex' / 'core').rglob('*.py'))
        for module in _imported_modules(path, '.'.join(path.relative_to(source).parent.parts))
        if module == SYSTEMS or module.startswith(f'{SYSTEMS}.')
    ]
    assert not found, 'core modules import the systems:\n' + '\n'.join(found)


def test_shared_packages_hold_only_subpackages(request):
    """Test that the roots of `systems/general` and of every field `common` hold only `__init__.py` and subpackages."""
    systems = request.config.rootpath / 'src' / 'skcomplex' / 'systems'
    roots = [systems / 'general', *sorted(path.parent for path in systems.glob('*/common/__init__.py'))]
    found = [
        str(path.relative_to(systems))
        for root in roots
        for path in sorted(root.glob('*.py'))
        if path.name != '__init__.py'
    ]
    assert not found, 'loose modules at the root of a shared package:\n' + '\n'.join(found)


def _public_classes() -> list[type]:
    """Return the public classes of the subpackages of `skcomplex.core.components`, without typed dictionaries."""
    found: dict[str, type] = {}
    for name in ('elements', 'interactions', 'laws', 'spaces', 'specs', 'systems'):
        package = importlib.import_module(f'skcomplex.core.components.{name}')
        for item in package.__all__:
            value = getattr(package, item)
            if isinstance(value, type) and not is_typeddict(value):
                found[item] = value
    return list(found.values())


def _named_classes(path: Path) -> tuple[set[str], set[str]]:
    """Return the names a core module uses in its code, and the names of the bases of the classes it defines."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    bases = {
        base.id
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        for base in node.bases
        if isinstance(base, ast.Name)
    }
    return names, bases


def _is_structural_base(klass: type, classes: list[type]) -> bool:
    """Return whether a class is a base, by its name or because another public class of the core subclasses it."""
    return klass.__name__.startswith('Base') or any(
        other is not klass and issubclass(other, klass) for other in classes
    )


def _is_handled_by_type(klass: type, modules: dict[Path, tuple[set[str], set[str]]]) -> bool:
    """Return whether a core module other than its own and those of its subclasses names a class in its code."""
    own = Path(str(importlib.import_module(klass.__module__).__file__))
    return any(
        klass.__name__ in names and klass.__name__ not in bases
        for path, (names, bases) in modules.items()
        if path != own
    )


def _is_general_form(klass: type, classes: list[type]) -> bool:
    """Return whether an interaction lists the fewest law classes among those of its base that list one of its laws.

    `RuleLaw` counts for no class, since a law of arbitrary code is never a mathematical operation the engine lowers.
    """
    if not issubclass(klass, BaseInteraction):
        return False
    laws = cast(type[BaseInteraction], klass)._get_law_classes()
    siblings = [
        cast(type[BaseInteraction], other)._get_law_classes()
        for other in classes
        if issubclass(other, BaseInteraction) and other.__base__ is klass.__base__
    ]
    return any(
        len(laws) == min(len(listed) for listed in siblings if law in listed) for law in laws if law is not RuleLaw
    )


def test_core_components_obey_the_placement_rule(request):
    """Test that every public class of the core components is a base, is handled by type, or is a law's general form."""
    core = request.config.rootpath / 'src' / 'skcomplex' / 'core'
    modules = {path: _named_classes(path) for path in sorted(core.rglob('*.py'))}
    classes = _public_classes()
    found = [
        klass.__name__
        for klass in classes
        if not (
            _is_structural_base(klass, classes)
            or _is_handled_by_type(klass, modules)
            or _is_general_form(klass, classes)
        )
    ]
    assert not found, 'core classes that are recipes:\n' + '\n'.join(sorted(found))
