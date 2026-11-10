"""Test the sources of the repository: the package, its tests, and its documentation hide no finding of the checks."""

import ast
import doctest
import importlib
import inspect
import multiprocessing
import re
import textwrap
import tokenize
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
CONCEPT_PLACEHOLDER = f'{GENERAL}.<concept>'
GENERATED = Path('docs') / 'generated'
FORBIDDEN = (
    'TYPE' + '_CHECKING',
    'no' + 'qa',
    'type' + ': ignore',
    'no' + 'sec',
)
CONSTANT_NAME = re.compile(r'_*[A-Z][A-Z0-9_]*')
IMPORTS = (ast.Import, ast.ImportFrom)
PROPERTIES = ('property', 'cached_property')
PACKAGE = 'skcomplex'
DUNDER = re.compile(r'__\w+__')
LIBRARY_HOOK = re.compile(r'_[a-z][a-z0-9_]*[a-z0-9]_')
FACTORY_KEYWORD = re.compile(r'\w+_func|latex_name')
UTILS_MODULE = re.compile(r'\._utils$|\.utils\.')
CORE_GUARD = Path('tests') / 'core' / 'components' / 'test_base.py'
FENCE = re.compile(r'^\s*(```|~~~)')
INLINE_CODE = re.compile('`[^`\n]*`')
REWRITTEN_DOCUMENTS: tuple[str, ...] = (
    'docs/overview/user_guide/build_a_custom_system/index.md',
    'docs/overview/user_guide/core_components/elements.md',
    'docs/overview/user_guide/core_components/index.md',
)
CORE_COMPONENTS = USER_GUIDE / 'core_components'
FAMILY_PAGES = {'elements.md': 'skcomplex.core.components.elements'}
GALLERY = Path('docs') / 'examples'
OWN_RECEIVERS = ('self', 'cls', 'super()')
DOCTESTS = doctest.DocTestParser()
BANNED_PHRASES = (
    'in summary',
    'in conclusion',
    'to summarize',
    'it is worth noting',
    "it's worth noting",
    'note that',
    "let's dive",
    'dive into',
    'delve',
    'leverage',
    'seamless',
    'seamlessly',
    'robust',
    'powerful',
    'cutting-edge',
    'unlock',
    'harness',
    'empower',
    'plethora',
    'myriad',
    'a wide range of',
    "in today's",
    'simply',
    'just',
    'easily',
    'magic',
)
VERBS = frozenset(
    [
        'accelerate',
        'accept',
        'accepts',
        'accumulate',
        'act',
        'activate',
        'add',
        'adjust',
        'adopt',
        'adopts',
        'advance',
        'align',
        'allocate',
        'append',
        'apply',
        'arrange',
        'arrive',
        'assemble',
        'assign',
        'attach',
        'average',
        'avoid',
        'balance',
        'barter',
        'bear',
        'begin',
        'bind',
        'binds',
        'borrow',
        'breaks',
        'breed',
        'broadcast',
        'build',
        'burn',
        'buy',
        'call',
        'carries',
        'carry',
        'catch',
        'change',
        'changes',
        'charge',
        'check',
        'choose',
        'claim',
        'clear',
        'climb',
        'close',
        'coerce',
        'collect',
        'collide',
        'combine',
        'combines',
        'commute',
        'compete',
        'compile',
        'compiles',
        'compute',
        'contains',
        'continues',
        'contract',
        'controls',
        'convert',
        'convolve',
        'copy',
        'count',
        'create',
        'cut',
        'decide',
        'declare',
        'declares',
        'decode',
        'defer',
        'deposit',
        'derive',
        'descend',
        'describe',
        'destine',
        'destroy',
        'die',
        'dies',
        'diffuse',
        'dispatch',
        'display',
        'disrupt',
        'divide',
        'draw',
        'drive',
        'drop',
        'duplicate',
        'embed',
        'emit',
        'encode',
        'enter',
        'enumerate',
        'evaluate',
        'evolve',
        'exchange',
        'excludes',
        'execute',
        'expand',
        'expect',
        'extend',
        'feed',
        'feel',
        'feels',
        'fight',
        'fill',
        'filter',
        'find',
        'finish',
        'fire',
        'fit',
        'fits',
        'fix',
        'flatten',
        'flip',
        'fly',
        'follow',
        'forecast',
        'forget',
        'fork',
        'format',
        'gain',
        'gather',
        'get',
        'give',
        'go',
        'grow',
        'handle',
        'happens',
        'harvest',
        'has',
        'herd',
        'hire',
        'hoist',
        'hold',
        'holds',
        'hop',
        'imitate',
        'increment',
        'infect',
        'infer',
        'influence',
        'initialize',
        'integrate',
        'intensify',
        'interact',
        'interpret',
        'invade',
        'invent',
        'is',
        'iterate',
        'join',
        'jump',
        'jumps',
        'kick',
        'land',
        'lays',
        'learn',
        'leave',
        'lend',
        'ligate',
        'link',
        'live',
        'load',
        'locate',
        'log',
        'lower',
        'make',
        'map',
        'mark',
        'mask',
        'match',
        'measure',
        'merge',
        'mirror',
        'move',
        'moves',
        'multiplies',
        'multiply',
        'mutate',
        'need',
        'negate',
        'nests',
        'note',
        'observe',
        'open',
        'operate',
        'pad',
        'pay',
        'parse',
        'partition',
        'pick',
        'place',
        'plan',
        'play',
        'pluck',
        'populate',
        'predict',
        'prepare',
        'print',
        'produce',
        'project',
        'proliferate',
        'propagate',
        'propose',
        'pull',
        'push',
        'quote',
        'raise',
        'rank',
        'react',
        'read',
        'reads',
        'rebuild',
        'record',
        'recount',
        'redesign',
        'redraw',
        'reduce',
        'regrow',
        'reject',
        'relax',
        'release',
        'remember',
        'remove',
        'rename',
        'renew',
        'replace',
        'replay',
        'replicate',
        'report',
        'reprice',
        'request',
        'reset',
        'resize',
        'resolve',
        'respond',
        'restart',
        'restock',
        'retarget',
        'returns',
        'rewire',
        'rewrite',
        'rotate',
        'run',
        'runs',
        'sample',
        'scale',
        'scatter',
        'schedule',
        'score',
        'scroll',
        'search',
        'searches',
        'secretes',
        'seed',
        'select',
        'sell',
        'separate',
        'set',
        'settle',
        'shift',
        'shop',
        'simulate',
        'smooth',
        'solve',
        'split',
        'spread',
        'spreads',
        'stalls',
        'start',
        'starves',
        'stay',
        'steer',
        'step',
        'stops',
        'strips',
        'substitute',
        'subtract',
        'sum',
        'sums',
        'supply',
        'support',
        'supports',
        'sweep',
        'swim',
        'swing',
        'switches',
        'tag',
        'take',
        'trade',
        'transfer',
        'translate',
        'transport',
        'travel',
        'triangulates',
        'truncate',
        'tune',
        'turn',
        'type',
        'unpack',
        'unroll',
        'update',
        'visit',
        'wait',
        'waits',
        'walk',
        'walks',
        'wander',
        'warm',
        'warn',
        'wire',
        'work',
        'wrap',
        'write',
    ],
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
    """Return whether a dotted path below the systems names a subfield, the general package, or a concept of it."""
    return path in (SUBFIELD_PLACEHOLDER, GENERAL, CONCEPT_PLACEHOLDER) or SUBFIELD_PACKAGE.fullmatch(path) is not None


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


def _prose_faults(line: str) -> list[str]:
    """Return the semicolons, dashes used as punctuation, and banned phrases of a prose line without inline code."""
    faults = []
    if ';' in line:
        faults.append('semicolon')
    if re.search('[\u2013\u2014]', line) or ' - ' in line.strip()[1:]:
        faults.append('dash')
    lowered = line.lower()
    faults += [phrase for phrase in BANNED_PHRASES if re.search(rf'(?<![\w-]){re.escape(phrase)}(?![\w-])', lowered)]
    return faults


def _markdown_prose(text: str) -> list[tuple[int, str]]:
    """Return the prose lines of a Markdown text with their numbers, leaving out fenced blocks and inline code."""
    lines = []
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            lines.append((number, INLINE_CODE.sub('', line)))
    return lines


def _example_prose(text: str) -> list[tuple[int, str]]:
    """Return the prose lines of a gallery example with their numbers: its docstring and its comments."""
    docstring = ast.get_docstring(ast.parse(text), clean=False) or ''
    lines = [(number + 1, line) for number, line in _markdown_prose(docstring)]
    first = len(docstring.splitlines()) + 2
    for number, line in enumerate(text.splitlines()[first:], start=first + 1):
        if line.lstrip().startswith('#') and not line.lstrip().startswith('# %%'):
            lines.append((number, INLINE_CODE.sub('', line.lstrip()[1:])))
    return lines


def _unnatural_lines(root: Path, documents: tuple[str, ...]) -> list[str]:
    """Return every prose line of the documents with a semicolon, a dash, or a banned phrase, as `file:line: faults`."""
    found = []
    for document in documents:
        text = (root / document).read_text(encoding='utf-8')
        prose = _example_prose(text) if document.endswith('.py') else _markdown_prose(text)
        found += [
            f'{document}:{number}: {", ".join(faults)}: {line.strip()}'
            for number, line in prose
            if (faults := _prose_faults(line))
        ]
    return found


def test_documents_read_naturally(request):
    """Test that the prose of the rewritten documents holds no semicolon, dash, or banned phrase."""
    found = _unnatural_lines(request.config.rootpath, REWRITTEN_DOCUMENTS)
    assert not found, f'{len(found)} lines that do not read naturally:\n' + '\n'.join(found)


def test_core_components_pages_cover_every_class(request):
    """Test that each family page has a section per public class and imports nothing from the catalog."""
    found = []
    for page, package in FAMILY_PAGES.items():
        path = request.config.rootpath / CORE_COMPONENTS / page
        headings = set(re.findall(r'^## (\w+)$', path.read_text(encoding='utf-8'), flags=re.MULTILINE))
        module = importlib.import_module(package)
        classes = [getattr(module, name) for name in module.__all__]
        found += [
            f'{page}: no section for {item.__name__}'
            for item in classes
            if isinstance(item, type) and not inspect.isabstract(item) and item.__name__ not in headings
        ]
        for start, block in _python_blocks(path):
            found += [
                f'{page}:{start + line}: imports {module_name}'
                for line, module_name, _ in _imported_names(block)
                if module_name == SYSTEMS or module_name.startswith(f'{SYSTEMS}.')
            ]
    assert not found, '\n'.join(found)


def test_banned_phrases_match_the_core_guard(request):
    """Test that the banned phrases of the documents equal those of the docstring guard of the core."""
    tree = ast.parse((request.config.rootpath / CORE_GUARD).read_text(encoding='utf-8'))
    values = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and [ast.unparse(target) for target in node.targets] == ['BANNED_PHRASES']
    ]
    assert values == [BANNED_PHRASES]


def _is_private(name: str) -> bool:
    """Return whether a name is private, leaving out dunder names and the throwaway name `_`."""
    return name.startswith('_') and name != '_' and DUNDER.fullmatch(name) is None


def _private_uses(code: str) -> list[tuple[int, str]]:
    """Return every call or attribute read of a private name in a block of code, as its line and its expression.

    A name that the block defines itself, such as a hook of a subclass or its own helper, may be read through `self`,
    `cls`, or `super()`, and a function it defines may be called.
    """
    tree = ast.parse(code)
    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)}
    nodes = list(ast.walk(tree))
    attributes = [
        (node.lineno, ast.unparse(node))
        for node in nodes
        if isinstance(node, ast.Attribute)
        and _is_private(node.attr)
        and not (node.attr in defined and ast.unparse(node.value) in OWN_RECEIVERS)
    ]
    calls = [
        (node.lineno, node.func.id)
        for node in nodes
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and _is_private(node.func.id)
        and node.func.id not in defined
    ]
    return sorted(attributes + calls)


def _docstring_examples(path: Path) -> list[tuple[int, str]]:
    """Return the doctest examples of the docstrings of a module, each with the line of the file before it."""
    examples = []
    for node in ast.walk(_read_tree(path)):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        docstring = ast.get_docstring(node, clean=False)
        if docstring and '>>>' in docstring:
            start = node.body[0].lineno - 1
            examples += [(start + example.lineno, example.source) for example in DOCTESTS.get_examples(docstring)]
    return examples


def _shown_private_names(root: Path) -> list[str]:
    """Return every private name that a docstring example, a document, or a gallery example calls or reads."""
    samples = [(path, *sample) for path in sorted((root / 'src').rglob('*.py')) for sample in _docstring_examples(path)]
    samples += [(root / document, *block) for document in _documents(root) for block in _python_blocks(root / document)]
    samples += [(path, 0, path.read_text(encoding='utf-8')) for path in sorted((root / GALLERY).rglob('*.py'))]
    return [
        f'{path.relative_to(root)}:{start + line}: {name}'
        for path, start, code in samples
        for line, name in _private_uses(code)
    ]


def test_documents_show_only_public_names(request):
    """Test that no docstring example, document, or gallery example calls or reads a private name of the library."""
    found = _shown_private_names(request.config.rootpath)
    assert not found, f'{len(found)} uses of private names:\n' + '\n'.join(found)


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


def _read_tree(path: Path) -> ast.Module:
    """Return the syntax tree of a Python file."""
    return ast.parse(path.read_text(encoding='utf-8'))


def _assigned_name(node: ast.stmt) -> str | None:
    """Return the one name a module assignment binds, or nothing for another statement."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _constant_name(node: ast.stmt) -> str | None:
    """Return the name a module constant binds, or nothing for another statement."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
        names = [item.id if isinstance(item, ast.Name) else '' for item in node.targets[0].elts]
        return ', '.join(names) if all(CONSTANT_NAME.fullmatch(name) for name in names) else None
    name = _assigned_name(node)
    return name if name is not None and CONSTANT_NAME.fullmatch(name) else None


def _read_names(node: ast.AST) -> set[str]:
    """Return the plain names an expression or a statement reads."""
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _is_docstring(node: ast.stmt) -> bool:
    """Return whether a statement is a bare string, as a docstring is."""
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _derived_indices(body: list[ast.stmt]) -> dict[int, int]:
    """Return the module assignments whose value reads a definition of the module, with the last statement they read.

    A derived value reads a class or a function of its module, or another derived value, so it cannot sit above it.
    """
    defined: dict[str, int] = {}
    derived: dict[int, int] = {}
    for index, node in enumerate(body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined[node.name] = index
            continue
        name = _assigned_name(node)
        value = getattr(node, 'value', None)
        read = [defined[read] for read in _read_names(value) if read in defined] if value is not None else []
        if name is not None and read:
            derived[index] = max(read)
            defined[name] = index
    return derived


def _misplaced_constants(path: Path) -> list[str]:
    """Return every module constant of a file that is private or does not lead its module, as `file:line: name`."""
    body = _read_tree(path).body
    last_import = max((index for index, node in enumerate(body) if isinstance(node, IMPORTS)), default=-1)
    derived = _derived_indices(body)
    found: list[str] = []
    for index, node in enumerate(body):
        name = _constant_name(node)
        if name is None:
            continue
        if name.startswith('_'):
            found.append(f'{path}:{node.lineno}: {name} is private')
        if index in derived:
            above = [other for other in range(derived[index] + 1, index) if other not in derived]
        else:
            above = [
                other
                for other in range(index)
                if not (
                    (other == 0 and _is_docstring(body[other]))
                    or other <= last_import
                    or (_constant_name(body[other]) is not None and other not in derived)
                )
            ]
        if above:
            found.append(f'{path}:{node.lineno}: {name} sits below line {body[above[0]].lineno}')
    return found


def _blank_and_comment_lines(path: Path) -> tuple[set[int], set[int]]:
    """Return the blank lines and the lines holding a comment of a Python file, read from its tokens."""
    blank: set[int] = set()
    comments: set[int] = set()
    with tokenize.open(path) as stream:
        previous = 0
        for token in tokenize.generate_tokens(stream.readline):
            if token.type == tokenize.COMMENT:
                comments.add(token.start[0])
            elif token.type == tokenize.NL and token.start[0] != previous:
                blank.add(token.start[0])
            if token.type not in (tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
                previous = token.end[0]
    return blank, comments


def _constant_gaps(path: Path) -> list[str]:
    """Return every gap of a block of module constants, as `file:line: what`."""
    body = _read_tree(path).body
    blank, comments = _blank_and_comment_lines(path)
    found: list[str] = []
    for index, node in enumerate(body):
        name = _constant_name(node)
        if name is None:
            continue
        follows = index > 0 and _constant_name(body[index - 1]) is not None
        start = (body[index - 1].end_lineno or node.lineno) + 1 if follows else node.lineno
        lines = range(start, (node.end_lineno or node.lineno) + 1)
        found += [f'{path}:{line}: blank line in the block of {name}' for line in lines if line in blank]
        found += [f'{path}:{line}: comment in the block of {name}' for line in lines if line in comments]
        if not follows and node.lineno - 1 in comments:
            found.append(f'{path}:{node.lineno - 1}: comment above the block of {name}')
        if index + 1 < len(body) and _is_docstring(body[index + 1]):
            found.append(f'{path}:{body[index + 1].lineno}: attribute docstring of {name}')
    return found


def _module_name(path: Path, source: Path) -> str:
    """Return the dotted name of a module of the sources."""
    parts = path.relative_to(source).with_suffix('').parts
    return '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)


def _imported_bindings(tree: ast.Module, module: str, is_package: bool) -> dict[str, str]:
    """Return the names a module binds by its imports, each with the dotted name it binds."""
    package = module if is_package else module.rpartition('.')[0]
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.partition('.')[0]] = (
                    alias.name if alias.asname else alias.name.partition('.')[0]
                )
        elif isinstance(node, ast.ImportFrom):
            base = package.split('.')[: len(package.split('.')) - node.level + 1] if node.level else []
            source = '.'.join([*base, *([node.module] if node.module else [])])
            for alias in node.names:
                bindings[alias.asname or alias.name] = f'{source}.{alias.name}'
    return bindings


def _is_compiled(node: ast.FunctionDef, bindings: dict[str, str]) -> bool:
    """Return whether a function carries a decorator of Numba, in any import form."""
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        while isinstance(target, ast.Attribute):
            target = target.value
        if isinstance(target, ast.Name) and bindings.get(target.id, '').split('.')[0] == 'numba':
            return True
    return False


def _read_targets(node: ast.AST, bindings: dict[str, str], module: str, local: set[str]) -> set[str]:
    """Return the dotted names of the module functions a statement reads, of its own module or imported."""
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            if child.id in local:
                found.add(f'{module}.{child.id}')
            elif child.id in bindings:
                found.add(bindings[child.id])
        elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name) and child.value.id in bindings:
            found.add(f'{bindings[child.value.id]}.{child.attr}')
    return found


def _function_readers(source: Path) -> tuple[dict[str, set[str]], dict[str, ast.FunctionDef], set[str], set[str]]:
    """Return the readers of every module function of the sources, its definition, the compiled ones, and `__all__`.

    A reader is named `module.Name` for a class or a function, and `module:line` for another module statement.
    """
    readers: dict[str, set[str]] = {}
    functions: dict[str, ast.FunctionDef] = {}
    compiled: set[str] = set()
    exported: set[str] = set()
    for path in sorted(source.rglob('*.py')):
        module = _module_name(path, source)
        tree = _read_tree(path)
        bindings = _imported_bindings(tree, module, path.name == '__init__.py')
        local = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                functions[f'{module}.{node.name}'] = node
                if _is_compiled(node, bindings):
                    compiled.add(f'{module}.{node.name}')
            if _assigned_name(node) == '__all__':
                exported |= {
                    item.value
                    for item in ast.walk(node)
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                }
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for part, reader in _reading_parts(node, module):
                for target in _read_targets(part, bindings, module, local):
                    if target != reader:
                        readers.setdefault(target, set()).add(reader)
    return readers, functions, compiled, exported


def _reading_parts(node: ast.stmt, module: str) -> list[tuple[ast.AST, str]]:
    """Return the parts of a module statement that read names, each with the reader it stands for.

    The methods of a class read as the class. Its other statements run when the class is built, so they read as a value,
    `module:line`, as a module statement does.
    """
    if isinstance(node, ast.FunctionDef):
        return [(node, f'{module}.{node.name}')]
    if not isinstance(node, ast.ClassDef):
        return [(node, f'{module}:{node.lineno}')]
    parts: list[tuple[ast.AST, str]] = []
    for child in node.body:
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            parts.append((child, f'{module}.{node.name}'))
        else:
            parts.append((child, f'{module}:{child.lineno}'))
    return parts


def _class_readers(name: str, readers: dict[str, set[str]], functions: dict[str, ast.FunctionDef]) -> set[str]:
    """Return the classes that read a module function, directly or through other private module functions."""
    found: set[str] = set()
    seen = {name}
    pending = [name]
    while pending:
        for reader in readers.get(pending.pop(), set()):
            if reader in seen or ':' in reader:
                continue
            seen.add(reader)
            if reader not in functions:
                found.add(reader)
            elif reader.rpartition('.')[2].startswith('_'):
                pending.append(reader)
    return found


def _is_value_read(name: str, readers: dict[str, set[str]], functions: dict[str, ast.FunctionDef]) -> bool:
    """Return whether a module value reads a module function, directly or through other private module functions."""
    seen = {name}
    pending = [name]
    while pending:
        for reader in readers.get(pending.pop(), set()):
            if ':' in reader:
                return True
            if reader in functions and reader not in seen and reader.rpartition('.')[2].startswith('_'):
                seen.add(reader)
                pending.append(reader)
    return False


def _unowned_functions(source: Path, packages: tuple[str, ...]) -> list[str]:
    """Return every module function of the packages that a class should own, or that no package re-exports."""
    readers, functions, compiled, exported = _function_readers(source)
    found: list[str] = []
    for name, node in functions.items():
        module, _, function = name.rpartition('.')
        if not module.startswith(packages) or name in compiled:
            continue
        if function.startswith('_'):
            classes = _class_readers(name, readers, functions)
            derived = _is_value_read(name, readers, functions)
            if classes and not derived:
                found.append(f'{module}:{node.lineno}: {function} is read by {", ".join(sorted(classes))}')
        elif '._' in f'.{module}' and function not in exported and not UTILS_MODULE.search(module):
            found.append(f'{module}:{node.lineno}: {function} is public in a private module and not re-exported')
    return found


def _is_property(node: ast.FunctionDef) -> bool:
    """Return whether a method is a property, or its setter or deleter."""
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id in PROPERTIES:
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr in ('setter', 'getter', 'deleter', *PROPERTIES):
            return True
    return False


def _foreign_names(module: str, qualname: str) -> set[str]:
    """Return the names that the bases of a class from another library define."""
    value: object = importlib.import_module(module)
    for part in qualname.split('.'):
        value = getattr(value, part, None)
    if not isinstance(value, type):
        return set()
    return {name for base in value.__mro__[1:] if not base.__module__.startswith(PACKAGE) for name in vars(base)}


def _methods(tree: ast.Module) -> list[tuple[str, ast.FunctionDef]]:
    """Return every method of a module, each with the qualified name of its class."""
    found: list[tuple[str, ast.FunctionDef]] = []
    pending: list[tuple[str, ast.stmt]] = [('', node) for node in tree.body]
    while pending:
        prefix, node = pending.pop(0)
        if isinstance(node, ast.ClassDef):
            qualname = f'{prefix}{node.name}'
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    found.append((qualname, child))
                elif isinstance(child, ast.ClassDef):
                    pending.append((f'{qualname}.', child))
    return found


def _is_exempt_method(name: str) -> bool:
    """Return whether a method name is a magic method or a hook a library calls by a fixed name, as `_eval_`."""
    return bool(DUNDER.fullmatch(name) or LIBRARY_HOOK.fullmatch(name))


def _nonverb_methods(path: Path, module: str) -> list[str]:
    """Return every method of a module whose name does not begin with a verb, as `file:line: Class.method`."""
    found: list[str] = []
    foreign: dict[str, set[str]] = {}
    for qualname, node in _methods(_read_tree(path)):
        if _is_exempt_method(node.name) or _is_property(node) or node.name.strip('_').split('_')[0] in VERBS:
            continue
        if qualname not in foreign:
            foreign[qualname] = _foreign_names(module, qualname)
        if node.name not in foreign[qualname]:
            found.append(f'{path}:{node.lineno}: {qualname}.{node.name}')
    return found


def _sage_function_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Return the names that bind Sage's function factory, and the names that bind a Sage module."""
    factories: set[str] = set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('sage'):
            factories |= {alias.asname or alias.name for alias in node.names if alias.name == 'function'}
        elif isinstance(node, ast.Import):
            modules |= {alias.asname or alias.name for alias in node.names if alias.name.startswith('sage')}
    return factories, modules


def _factory_calls(path: Path) -> list[str]:
    """Return every call of Sage's function factory that defines a library function, as `file:line`."""
    tree = _read_tree(path)
    factories, modules = _sage_function_names(tree)
    found: list[str] = []
    for statement in tree.body:
        top = not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            is_factory = (isinstance(target, ast.Name) and target.id in factories) or (
                isinstance(target, ast.Attribute)
                and target.attr == 'function'
                and isinstance(target.value, ast.Name)
                and target.value.id in modules
            )
            hooks = [keyword.arg for keyword in node.keywords if keyword.arg and FACTORY_KEYWORD.fullmatch(keyword.arg)]
            if is_factory and (top or hooks):
                found.append(f'{path}:{node.lineno}: function factory {"at module level" if top else ", ".join(hooks)}')
    return found


def _python_files(folders: tuple[str, ...], root: Path) -> list[Path]:
    """Return the Python files of the checked folders, without the generated pages."""
    return [
        path
        for folder in folders
        for path in sorted((root / folder).rglob('*.py'))
        if not path.relative_to(root).is_relative_to(GENERATED)
    ]


def _checked_modules(packages: tuple[str, ...], root: Path) -> list[tuple[Path, str]]:
    """Return the modules of the checked packages, each with its dotted name."""
    source = root / 'src'
    folders = tuple(str(Path('src', *package.split('.'))) for package in packages)
    return [(path, _module_name(path, source)) for path in _python_files(folders, root)]


def test_module_constants_lead_their_module(request):
    """Test that every module constant is public and sits in the block after the imports, or after what it reads."""
    found = [line for path in _python_files(FOLDERS, request.config.rootpath) for line in _misplaced_constants(path)]
    assert not found, 'misplaced constants:\n' + '\n'.join(found)


def test_constant_blocks_hold_no_gap(request):
    """Test that no blank line, comment, or attribute docstring sits in or right above a block of constants."""
    found = [line for path in _python_files(FOLDERS, request.config.rootpath) for line in _constant_gaps(path)]
    assert not found, 'gaps in the constant blocks:\n' + '\n'.join(found)


def test_classes_own_the_functions_they_read(request):
    """Test that every module function a class reads is a method, apart from kernels and helpers of public functions."""
    found = _unowned_functions(request.config.rootpath / 'src', (PACKAGE,))
    assert not found, 'module functions that a class should own:\n' + '\n'.join(found)


def test_methods_begin_with_a_verb(request):
    """Test that every method begins with a verb, apart from magic methods, library hooks, and properties."""
    modules = _checked_modules((PACKAGE,), request.config.rootpath)
    found = [line for path, module in modules for line in _nonverb_methods(path, module)]
    assert not found, 'methods that do not begin with a verb:\n' + '\n'.join(found)


def test_symbolic_functions_are_builtin_subclasses(request):
    """Test that Sage's function factory only declares unknown functions inside a function."""
    modules = _checked_modules((PACKAGE,), request.config.rootpath)
    found = [line for path, _ in modules for line in _factory_calls(path)]
    assert not found, 'symbolic functions built by the factory:\n' + '\n'.join(found)


def _write_module(folder: Path, name: str, code: str) -> Path:
    """Write a module of code to a folder and return its path."""
    path = folder / f'{name}.py'
    path.write_text(textwrap.dedent(code), encoding='utf-8')
    return path


def test_misplaced_constants_are_named(tmp_path):
    """Test that a private constant and a constant below a function are named, and a derived value is not."""
    path = _write_module(
        tmp_path,
        'constants',
        """
        import re
        LIMIT = 3
        def double(value):
            return 2 * value
        TWICE = double(LIMIT)
        _HIDDEN = 1
        """,
    )
    assert _misplaced_constants(path) == [
        f'{path}:7: _HIDDEN is private',
        f'{path}:7: _HIDDEN sits below line 4',
    ]


def test_constant_gaps_are_named(tmp_path):
    """Test that a blank line, a comment, and an attribute docstring of a block are named, and a tight block is not."""
    path = _write_module(
        tmp_path,
        'gaps',
        """
        LIMIT = 3

        # The size.
        SIZE = 4
        \"\"\"The size.\"\"\"
        """,
    )
    tight = _write_module(tmp_path, 'tight', 'LIMIT = 3\nSIZE = (\n    4,\n)\n')
    assert _constant_gaps(path) == [
        f'{path}:3: blank line in the block of SIZE',
        f'{path}:4: comment in the block of SIZE',
        f'{path}:6: attribute docstring of SIZE',
    ]
    assert _constant_gaps(tight) == []


def test_unowned_functions_are_named(tmp_path):
    """Test that a helper of a class and an unexported public function are named, and kernels and others are not."""
    package = tmp_path / 'pack'
    package.mkdir()
    _write_module(package, '__init__', "__all__ = ['run']")
    _write_module(
        package,
        '_module',
        """
        from numba import njit
        @njit
        def _kernel(value):
            return value
        def _helper(value):
            return _kernel(value)
        def _of_run(value):
            return value
        def _of_table(value):
            return value
        TABLE = {'one': _of_table}
        def run(value):
            return _of_run(value)
        def spare(value):
            return value
        class Model:
            def fit(self, value):
                return _helper(value) + TABLE['one'](value)
        """,
    )
    found = _unowned_functions(tmp_path, ('pack',))
    assert found == [
        'pack._module:6: _helper is read by pack._module.Model',
        'pack._module:15: spare is public in a private module and not re-exported',
    ]


def test_nonverb_methods_are_named(tmp_path, monkeypatch):
    """Test that a method that begins with a noun is named, and verbs, hooks, properties, and dunders are not."""
    monkeypatch.syspath_prepend(str(tmp_path))
    path = _write_module(
        tmp_path,
        'methods',
        """
        class Model:
            def __init__(self):
                self.size = 1
            def _eval_(self):
                return 0
            @property
            def total(self):
                return 0
            def get_total(self):
                return 0
            def _noise(self):
                return 0
        """,
    )
    assert _nonverb_methods(path, 'methods') == [f'{path}:12: Model._noise']


def test_factory_calls_are_named(tmp_path):
    """Test that a factory call at module level or with a hook is named, and an unknown inside a function is not."""
    path = _write_module(
        tmp_path,
        'factories',
        """
        from sage.all__sagemath_symbolics import function as make
        LAG = make('lag', nargs=2)
        def build():
            return make('lead', print_func=str), make('a')
        """,
    )
    assert _factory_calls(path) == [
        f'{path}:3: function factory at module level',
        f'{path}:5: function factory print_func',
    ]


def test_unnatural_lines_are_named(tmp_path):
    """Test that a dash, a semicolon, and a banned phrase in prose are named, and code and list markers are not."""
    (tmp_path / 'page.md').write_text(
        textwrap.dedent(
            """
            # Page

            - A list item stays allowed, and so does `a - b` in inline code.
            The text runs on \u2014 and on.
            It is a test - nothing more; it is robust.

            ```python
            total = 1 - 2  # a dash in code
            ```
            """,
        ),
        encoding='utf-8',
    )
    example = '"""\n# Example\n\nThe text runs on - and on.\n"""\n\n# %%\n# Note that it ends.\n'
    _write_module(tmp_path, 'plot_example', example)
    assert _unnatural_lines(tmp_path, ('page.md', 'plot_example.py')) == [
        'page.md:5: dash: The text runs on \u2014 and on.',
        'page.md:6: semicolon, dash, robust: It is a test - nothing more; it is robust.',
        'plot_example.py:5: dash: The text runs on - and on.',
        'plot_example.py:8: note that: Note that it ends.',
    ]


def test_private_names_are_named(tmp_path):
    """Test that a call of a private hook is named, and a subclass that defines and extends the hook is not."""
    code = textwrap.dedent(
        """
        class QuietElement(Element):
            def _get_laws(self, namespace):
                return [*super()._get_laws(namespace), *self._get_extra(namespace)]

            def _get_extra(self, namespace):
                return []


        system._get_laws(namespace)
        system._parts
        """,
    )
    assert _private_uses(code) == [(10, 'system._get_laws'), (11, 'system._parts')]
