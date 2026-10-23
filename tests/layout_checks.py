"""Check the layout rules of the core with the syntax tree, each check naming the module and the offending name.

Every check takes the source root, the directory that holds the `skcomplex` package, so the tests run it on the real
tree and on small packages they plant in a temporary directory. Each returns the list of violations as messages.
"""

import ast
import re
from collections.abc import Collection, Iterable, Iterator
from pathlib import Path

CORE_ORDER = (
    'base',
    'space',
    'laws',
    'components',
    'interactions',
    'system',
    'compile',
    'solvers',
    'backends',
    'runtime',
    'viz',
    'experiments',
    'analysis',
    'formalisms',
)
CORE = 'skcomplex.core'
FORMALISMS = f'{CORE}.formalisms'
BASE = f'{CORE}.base'
BASE_CLASSES = frozenset({'Component', 'Law'})
KINDS = ('private function', 'public function', 'private class', 'public class')
DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
CLIENT_SUFFIXES = frozenset({'.py', '.md'})


def _module_of(path: Path, root: Path) -> tuple[str, bool]:
    """Return the dotted module of a source file under the root and whether it is a package's `__init__`."""
    parts = list(path.relative_to(root).with_suffix('').parts)
    if parts[-1] == '__init__':
        return '.'.join(parts[:-1]), True
    return '.'.join(parts), False


def _sources(root: Path, package: str = CORE) -> Iterator[tuple[str, bool, ast.Module]]:
    """Yield the module, whether it is a package, and the syntax tree of every source file of a package."""
    directory = root.joinpath(*package.split('.'))
    for path in sorted(directory.rglob('*.py')):
        module, is_package = _module_of(path, root)
        yield module, is_package, ast.parse(path.read_text())


def _imports(tree: ast.AST, module: str, *, is_package: bool) -> list[tuple[str, list[str], ast.stmt]]:
    """Return every import of a syntax tree as the absolute module, the names imported from it, and the statement."""
    package = module if is_package else module.rpartition('.')[0]
    found: list[tuple[str, list[str], ast.stmt]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name, [], node) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split('.')
                base = base[: len(base) - (node.level - 1)]
                target = '.'.join([*base, *([node.module] if node.module else [])])
            else:
                target = node.module or ''
            found.append((target, [alias.name for alias in node.names], node))
    return found


def _reached(target: str, names: list[str]) -> list[str]:
    """Return the modules an import reaches: its module, and its module joined with each name it imports."""
    return [target, *(f'{target}.{name}' for name in names)]


def _core_package(module: str) -> str | None:
    """Return the core package a module belongs to, or nothing for a module outside the core's packages."""
    parts = module.split('.')
    return parts[2] if parts[:2] == CORE.split('.') and len(parts) > 2 else None  # noqa: PLR2004


def _formalism_of(module: str, formalisms: Iterable[str]) -> str | None:
    """Return the formalism a module belongs to, or nothing for a module outside every formalism."""
    parts = module.split('.')
    if module.startswith(f'{FORMALISMS}.') and parts[3] in formalisms:
        return parts[3]
    return None


def _formalism_names(root: Path) -> list[str]:
    """Return the formalisms of the tree, the subpackages of `skcomplex.core.formalisms`."""
    directory = root.joinpath(*FORMALISMS.split('.'))
    if not directory.is_dir():
        return []
    return sorted(path.parent.name for path in directory.glob('*/__init__.py'))


def _names(node: ast.AST | None) -> set[str]:
    """Return the names a node reads."""
    if node is None:
        return set()
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _signature_names(node: ast.FunctionDef | ast.AsyncFunctionDef, *, annotations: bool) -> set[str]:
    """Return the names a function's definition reads when it runs: its decorators, defaults, and annotations."""
    arguments = node.args
    found = set().union(*map(_names, [*node.decorator_list, *arguments.defaults, *arguments.kw_defaults]))
    if annotations:
        every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, arguments.vararg, arguments.kwarg]
        found |= set().union(*(_names(item.annotation) for item in every if item is not None), _names(node.returns))
    return found


def _import_time_names(node: ast.stmt, *, annotations: bool) -> set[str]:
    """Return the names a statement reads when the module runs, leaving out the bodies of functions."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return _signature_names(node, annotations=annotations)
    if isinstance(node, ast.ClassDef):
        found = set().union(*map(_names, [*node.decorator_list, *node.bases, *(item.value for item in node.keywords)]))
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and not annotations:
                found |= _names(item.value)
            else:
                found |= _import_time_names(item, annotations=annotations)
        return found
    return _names(node)


def _rank(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    """Return the turn of a definition: private functions, public functions, private classes, then public classes."""
    return 2 * isinstance(node, ast.ClassDef) + (not node.name.startswith('_'))


def _needed_early(tree: ast.Module) -> set[str]:
    """Return the definitions the module needs before their turn, with the definitions they need in turn.

    A definition is needed early when a definition of an earlier turn reads it as a base class, decorator, default
    value, or annotation, or when a module-level statement between the definitions reads it.
    """
    annotations = not any(
        isinstance(node, ast.ImportFrom)
        and node.module == '__future__'
        and any(alias.name == 'annotations' for alias in node.names)
        for node in tree.body
    )
    definitions = {node.name: node for node in tree.body if isinstance(node, DEFINITIONS)}
    last = max((index for index, node in enumerate(tree.body) if isinstance(node, DEFINITIONS)), default=-1)
    early: set[str] = set()
    for index, node in enumerate(tree.body):
        read = _import_time_names(node, annotations=annotations) & definitions.keys()
        if isinstance(node, DEFINITIONS):
            early |= {name for name in read if _rank(definitions[name]) > _rank(node)}
        elif index < last:
            early |= read
    pending = list(early)
    while pending:
        needed = _names(definitions[pending.pop()]) & definitions.keys()
        pending += needed - early
        early |= needed
    return early


def misordered_definitions(source: str, module: str) -> list[str]:
    """Return the definitions of a module that come after a definition of a later turn.

    The turns are private functions, public functions, private classes, then public classes. A definition the module
    needs before its turn may come early, and then it does not push the later definitions back.

    Args:
        source:
            The source of the module.

        module:
            The dotted name of the module, which the messages name.

    Returns:
        violations:
            One message per misplaced definition, naming the module, the definition, and the one it follows.
    """
    tree = ast.parse(source)
    early = _needed_early(tree)
    found, latest = [], None
    for node in (node for node in tree.body if isinstance(node, DEFINITIONS)):
        if latest is not None and _rank(node) < _rank(latest):
            kind, previous = KINDS[_rank(node)], KINDS[_rank(latest)]
            found.append(f'{module}: the {kind} {node.name} comes after the {previous} {latest.name}')
        if node.name not in early and (latest is None or _rank(node) > _rank(latest)):
            latest = node
    return found


def upward_imports(root: Path) -> list[str]:
    """Return the imports of a core package that reach a package listed after it in `CORE_ORDER`.

    Args:
        root:
            The directory that holds the `skcomplex` package.

    Returns:
        violations:
            One message per import, naming the module and the package it reached.
    """
    found = []
    for module, is_package, tree in _sources(root):
        package = _core_package(module)
        if package not in CORE_ORDER:
            continue
        for target, names, _ in _imports(tree, module, is_package=is_package):
            reached = {item for item in map(_core_package, _reached(target, names)) if item in CORE_ORDER}
            found += [
                f'{module}: {package} imports {CORE}.{item}, which comes after it'
                for item in sorted(reached)
                if CORE_ORDER.index(item) > CORE_ORDER.index(package)
            ]
    return sorted(set(found))


def definition_order(root: Path) -> list[str]:
    """Return the misplaced definitions of every module of the core, as `misordered_definitions` finds them.

    Args:
        root:
            The directory that holds the `skcomplex` package.

    Returns:
        violations:
            One message per misplaced definition.
    """
    directory = root.joinpath(*CORE.split('.'))
    return [
        item
        for path in sorted(directory.rglob('*.py'))
        for item in misordered_definitions(path.read_text(), _module_of(path, root)[0])
    ]


def wide_base_imports(root: Path) -> list[str]:
    """Return the first-party imports outside `skcomplex.core.base` of the modules that define a base class.

    The base classes are those of `BASE_CLASSES`, `Component` and `Law`.

    Args:
        root:
            The directory that holds the `skcomplex` package.

    Returns:
        violations:
            One message per import, naming the module and the module it imports.
    """
    found = []
    for module, is_package, tree in _sources(root):
        defined = sorted(node.name for node in tree.body if isinstance(node, ast.ClassDef)) if not is_package else []
        bases = [name for name in defined if name in BASE_CLASSES]
        if not bases:
            continue
        for target, _, _ in _imports(tree, module, is_package=is_package):
            first_party = target == 'skcomplex' or target.startswith('skcomplex.')
            if first_party and target != BASE and not target.startswith(f'{BASE}.'):
                found.append(f'{module}: defines the base {", ".join(bases)} and imports {target}, not only {BASE}')
    return found


def _is_type_checking(node: ast.expr) -> bool:
    """Check whether a condition is `TYPE_CHECKING`, bare or as an attribute."""
    return (isinstance(node, ast.Name) and node.id == 'TYPE_CHECKING') or (
        isinstance(node, ast.Attribute) and node.attr == 'TYPE_CHECKING'
    )


def guarded_imports(root: Path) -> list[str]:
    """Return the imports of the core under `if TYPE_CHECKING:`.

    Args:
        root:
            The directory that holds the `skcomplex` package.

    Returns:
        violations:
            One message per import, naming the module, its line, and the module it imports.
    """
    found = []
    for module, is_package, tree in _sources(root):
        for node in ast.walk(tree):
            if not (isinstance(node, ast.If) and _is_type_checking(node.test)):
                continue
            guarded = ast.Module(body=node.body, type_ignores=[])
            found += [
                f'{module}:{statement.lineno}: imports {target} under TYPE_CHECKING'
                for target, _, statement in _imports(guarded, module, is_package=is_package)
            ]
    return found


def formalism_imports(root: Path) -> list[str]:
    """Return the imports of `skcomplex.core.formalisms` by the shared core, every core module outside it.

    Args:
        root:
            The directory that holds the `skcomplex` package.

    Returns:
        violations:
            One message per import, naming the module and the module it imports.
    """
    found = []
    for module, is_package, tree in _sources(root):
        if module == FORMALISMS or module.startswith(f'{FORMALISMS}.'):
            continue
        for target, names, _ in _imports(tree, module, is_package=is_package):
            reached = [item for item in _reached(target, names) if f'{item}.'.startswith(f'{FORMALISMS}.')]
            found += [f'{module}: the shared core imports {reached[0]}'] if reached else []
    return found


def _declared_dependencies(root: Path, formalism: str) -> tuple[str, ...]:
    """Return the dependencies a formalism declares in its `Formalism(...)` call, read from its source.

    The `__init__` is read first, then the other modules of the formalism. A formalism without a declaration, or with
    dependencies that are not a literal, declares none.
    """
    directory = root.joinpath(*FORMALISMS.split('.'), formalism)
    paths = [directory / '__init__.py', *sorted(set(directory.rglob('*.py')) - {directory / '__init__.py'})]
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = function.attr if isinstance(function, ast.Attribute) else getattr(function, 'id', None)
            if name != 'Formalism':
                continue
            value = next((item.value for item in node.keywords if item.arg == 'depends'), None)
            value = node.args[1] if value is None and len(node.args) > 1 else value
            try:
                return tuple(ast.literal_eval(value)) if value is not None else ()
            except ValueError:
                return ()
    return ()


def undeclared_formalism_imports(root: Path) -> list[str]:
    """Return the imports of a formalism that reach another formalism it does not list in its `depends`.

    Args:
        root:
            The directory that holds the `skcomplex` package.

    Returns:
        violations:
            One message per formalism import, naming the module and both formalisms. No formalisms, no violations.
    """
    formalisms = _formalism_names(root)
    found = []
    for formalism in formalisms:
        declared = _declared_dependencies(root, formalism)
        for module, is_package, tree in _sources(root, f'{FORMALISMS}.{formalism}'):
            for target, names, _ in _imports(tree, module, is_package=is_package):
                reached = {_formalism_of(item, formalisms) or formalism for item in _reached(target, names)}
                found += [
                    f'{module}: the formalism {formalism} imports the formalism {other}, which it does not declare'
                    for other in sorted(reached - {formalism, *declared})
                ]
    return sorted(set(found))


def _exports(root: Path, package: str) -> list[str]:
    """Return the names a package's `__init__` lists in its `__all__`."""
    path = root.joinpath(*package.split('.'), '__init__.py')
    if not path.exists():
        return []
    value: ast.expr | None
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if value is not None and any(isinstance(target, ast.Name) and target.id == '__all__' for target in targets):
            return list(ast.literal_eval(value))
    return []


def _user_package(module: str) -> str | None:
    """Return the core package a module uses names as, where each formalism counts as its own package."""
    package = _core_package(module)
    parts = module.split('.')
    if package == 'formalisms' and len(parts) > 3 and not parts[3].startswith('_'):  # noqa: PLR2004
        return f'formalisms.{parts[3]}'
    return package


def _client_words(root: Path, clients: Iterable[Path]) -> set[str]:
    """Return the words of the models and of the client files and directories, the tests and the documentation."""
    paths = [*root.joinpath('skcomplex', 'models').rglob('*.py')]
    for client in clients:
        paths += [client] if client.is_file() else [path for path in client.rglob('*') if path.is_file()]
    words: set[str] = set()
    for path in paths:
        if path.suffix in CLIENT_SUFFIXES:
            words |= set(re.findall(r'[A-Za-z_]\w*', path.read_text(errors='ignore')))
    return words


def leaking_exports(root: Path, clients: Iterable[Path], concepts: Collection[str] = ()) -> list[str]:
    """Return the names a core package exports that one other core package alone uses, and no client.

    A core module uses a name when it imports it from the package or one of its modules. The clients are the models,
    the given files and directories, which are the tests and the documentation, and a client uses a name when its text
    holds it as a word. A concept of the package, which another package may use alone, is not a leak.

    Args:
        root:
            The directory that holds the `skcomplex` package.

        clients:
            The files and directories of the tests and the documentation.

        concepts:
            The qualified names, such as `skcomplex.core.base.SkcomplexError`, that are concepts of their package.

    Returns:
        violations:
            One message per leaking name, naming the package, the name, and the package that uses it.
    """
    users: dict[tuple[str, str], set[str]] = {}
    for module, is_package, tree in _sources(root):
        user = _user_package(module)
        for target, names, _ in _imports(tree, module, is_package=is_package):
            owner = _core_package(target)
            if owner is None or owner == _core_package(module):
                continue
            for name in names:
                users.setdefault((owner, name), set()).add(user or CORE)
    words = _client_words(root, clients)
    found = []
    for package in CORE_ORDER:
        for name in _exports(root, f'{CORE}.{package}'):
            using = users.get((package, name), set())
            concept = f'{CORE}.{package}.{name}' in concepts
            if len(using) == 1 and name not in words and not concept:
                found.append(f'{CORE}.{package}: exports {name}, which only {CORE}.{next(iter(using))} uses')
    return found
