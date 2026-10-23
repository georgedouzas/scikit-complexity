"""Rewrite `from X import a, b` statements after names or modules move, from tables of old and new import paths.

A table is a TOML or JSON file with a list `moves`. Each move has an `old` module and a `new` module, and optionally
the `names` that move. Without `names`, the whole module moves, with its submodules:

```toml
[[moves]]
old = "skcomplex.core.laws"
new = "skcomplex.core.formalisms.topology"
names = ["TopologyRequest", "add_edge"]

[[moves]]
old = "skcomplex.core.compile._topology"
new = "skcomplex.core.formalisms.topology._passes"
```

The script rewrites the import statements of Python files, the doctest lines of their docstrings, and the Python code
blocks and doctest lines of Markdown files. A statement whose names move to different modules splits into one
statement per module, and statements of one module at the top of a file merge into one. Relative imports stay relative.
Python files are formatted with black after the rewrite. Run `ruff check --select I --fix` afterwards to sort imports.

Usage: `python tools/rewrite_imports.py --table tools/import_moves src/skcomplex/models docs/examples tests`
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import argparse
import ast
import json
import re
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import black

LINE_LENGTH = 120
INDENT = '    '
DOCTEST_IMPORT = re.compile(r'^(?P<indent>[ \t]*)>>> from (?P<module>[\w.]+) import (?P<names>[\w, ]+?)[ \t]*$')
FENCE = re.compile(r'^(?P<indent>[ \t]*)```(?P<language>[\w-]*)[ \t]*$')
PYTHON_FENCES = frozenset({'python', 'py'})
DOCTEST_FENCES = frozenset({'pycon'})


@dataclass(frozen=True)
class Move:
    """Move of names, or of a whole module, from an old module to a new one.

    Args:
        old:
            The dotted path of the module the names leave.

        new:
            The dotted path of the module the names go to.

        names:
            The names that move, or `None` when the whole module moves with its submodules.
    """

    old: str
    new: str
    names: frozenset[str] | None = None

    def destination(self, module: str, name: str) -> str | None:
        """Return the module a name imported from a module moves to, or `None` when the move does not apply.

        Args:
            module:
                The absolute module the name is imported from.

            name:
                The imported name.

        Returns:
            destination:
                The new module, or `None`.
        """
        if self.names is not None:
            return self.new if module == self.old and name in self.names else None
        if module == self.old or module.startswith(f'{self.old}.'):
            return f'{self.new}{module[len(self.old) :]}'
        return None


@dataclass(frozen=True)
class _Group:
    """Names imported from one module, as written in one statement."""

    module: str
    aliases: tuple[str, ...]


def _destination(moves: Sequence[Move], module: str, name: str) -> str:
    """Return the module a name imported from a module comes from after the moves."""
    for move in moves:
        found = move.destination(module, name)
        if found is not None:
            return found
    return module


def _absolute(module: str | None, level: int, package: str | None) -> str | None:
    """Return the absolute module of an import, or `None` for a relative import outside a known package."""
    if not level:
        return module or ''
    if package is None:
        return None
    parts = package.split('.')
    if level - 1 >= len(parts):
        return None
    base = parts[: len(parts) - (level - 1)]
    return '.'.join([*base, *([module] if module else [])])


def _relative(target: str, package: str) -> str:
    """Return the relative form of an absolute module as seen from a package, or the module when unrelated."""
    target_parts, package_parts = target.split('.'), package.split('.')
    common = 0
    while common < min(len(target_parts), len(package_parts)) and target_parts[common] == package_parts[common]:
        common += 1
    if not common:
        return target
    level = len(package_parts) - common + 1
    return '.' * level + '.'.join(target_parts[common:])


def _alias_text(alias: ast.alias) -> str:
    """Return the text of one imported name with its alias."""
    return alias.name if alias.asname is None else f'{alias.name} as {alias.asname}'


def _groups(node: ast.ImportFrom, moves: Sequence[Move], package: str | None) -> list[_Group] | None:
    """Return the statements an import becomes after the moves, or `None` when no name of it moves."""
    module = _absolute(node.module, node.level, package)
    if module is None:
        return None
    destinations: dict[str, list[str]] = {}
    for alias in node.names:
        destinations.setdefault(_destination(moves, module, alias.name), []).append(_alias_text(alias))
    if list(destinations) == [module]:
        return None
    relative = node.level > 0 and package is not None
    return [
        _Group(_relative(target, package) if relative and package is not None else target, tuple(aliases))
        for target, aliases in destinations.items()
    ]


def _render(group: _Group, indent: str = '', prefix: str = '', continuation: str = '') -> list[str]:
    """Return the lines of one import statement, on one line when it fits, else one name per line."""
    line = f'{indent}{prefix}from {group.module} import {", ".join(group.aliases)}'
    if len(line) <= LINE_LENGTH:
        return [line]
    inner = f'{indent}{continuation}{INDENT}'
    return [
        f'{indent}{prefix}from {group.module} import (',
        *(f'{inner}{alias},' for alias in group.aliases),
        f'{indent}{continuation})',
    ]


def _comment(lines: list[str], node: ast.ImportFrom) -> str:
    """Return the comment that follows an import statement on its last line, or an empty string."""
    end_line, end_column = node.end_lineno, node.end_col_offset
    if end_line is None or end_column is None:
        return ''
    rest = lines[end_line - 1][end_column:].strip()
    return f'  {rest}' if rest.startswith('#') else ''


def _merged(statements: list[tuple[ast.ImportFrom, list[_Group], bool]]) -> dict[int, list[_Group]]:
    """Return the statements every import becomes, merging the top-level imports of a module a rewrite touches."""
    touched = {
        group.module for node, groups, changed in statements if changed and not node.col_offset for group in groups
    }
    first: dict[str, int] = {}
    merged: dict[int, list[_Group]] = {}
    for node, groups, _ in statements:
        kept = []
        for group in groups:
            if node.col_offset or group.module not in touched:
                kept.append(group)
                continue
            if group.module not in first:
                first[group.module] = id(node)
                kept.append(group)
                continue
            owner = merged[first[group.module]]
            index = next(index for index, item in enumerate(owner) if item.module == group.module)
            owner[index] = _Group(group.module, tuple(dict.fromkeys([*owner[index].aliases, *group.aliases])))
        merged[id(node)] = kept
    return merged


def _statements(
    tree: ast.Module,
    moves: Sequence[Move],
    package: str | None,
) -> list[tuple[ast.ImportFrom, list[_Group], bool]]:
    """Return every import of a tree, in source order, with the statements it becomes and whether it changes."""
    found = []
    nodes = sorted((node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)), key=lambda node: node.lineno)
    for node in nodes:
        groups = _groups(node, moves, package)
        own = [_Group('.' * node.level + (node.module or ''), tuple(_alias_text(item) for item in node.names))]
        found.append((node, own if groups is None else groups, groups is not None))
    return found if any(changed for _, _, changed in found) else []


def _rewrite_statements(source: str, moves: Sequence[Move], package: str | None) -> str:
    """Return Python source whose import statements follow the moves."""
    tree = ast.parse(source)
    statements = _statements(tree, moves, package)
    if not statements:
        return source
    lines = source.splitlines(keepends=True)
    plain = source.splitlines()
    merged = _merged(statements)
    for node, groups, changed in reversed(statements):
        if not changed and merged[id(node)] == groups:
            continue
        indent = ' ' * node.col_offset
        comment = _comment(plain, node)
        text = [line + comment for group in merged[id(node)] for line in _render(group, indent)]
        start, end = node.lineno - 1, node.end_lineno or node.lineno
        lines[start:end] = [f'{line}\n' for line in text]
    return ''.join(lines)


def _rewrite_doctests(text: str, moves: Sequence[Move]) -> str:
    """Return text whose single-line doctest imports follow the moves."""
    output = []
    for line in text.splitlines(keepends=True):
        found = DOCTEST_IMPORT.match(line.rstrip('\n'))
        if found is None:
            output.append(line)
            continue
        node = ast.parse(f'from {found["module"]} import {found["names"]}').body[0]
        groups = _groups(node, moves, None) if isinstance(node, ast.ImportFrom) else None
        if groups is None:
            output.append(line)
            continue
        indent = found['indent']
        output += [f'{rendered}\n' for group in groups for rendered in _render(group, indent, '>>> ', '... ')]
    return ''.join(output)


def _rewrite_markdown(text: str, moves: Sequence[Move]) -> str:
    """Return Markdown whose Python code blocks, console blocks, and doctest lines follow the moves."""
    output: list[str] = []
    block: list[str] | None = None
    language = ''
    for line in text.splitlines(keepends=True):
        fence = FENCE.match(line.rstrip('\n'))
        if block is None:
            output.append(line)
            if fence is not None and fence['language'] in PYTHON_FENCES | DOCTEST_FENCES:
                block, language = [], fence['language']
            continue
        if fence is None:
            block.append(line)
            continue
        code = ''.join(block)
        if language in PYTHON_FENCES:
            code = _rewrite_statements(code, moves, None)
        output += [_rewrite_doctests(code, moves), line]
        block = None
    output += block or []
    return ''.join(output)


def _package_of(path: Path) -> str | None:
    """Return the dotted package of a Python file, found from the `__init__.py` files above it."""
    parts = []
    directory = path.parent
    while (directory / '__init__.py').exists():
        parts.append(directory.name)
        directory = directory.parent
    return '.'.join(reversed(parts)) or None


def _black_mode(pyproject: Path) -> black.Mode:
    """Return the black mode the project configures."""
    settings = tomllib.loads(pyproject.read_text()).get('tool', {}).get('black', {}) if pyproject.exists() else {}
    return black.Mode(
        line_length=settings.get('line-length', LINE_LENGTH),
        string_normalization=not settings.get('skip-string-normalization', False),
    )


def _files(paths: Iterable[Path]) -> list[Path]:
    """Return the Python and Markdown files of paths, searching directories recursively."""
    found = []
    for path in paths:
        candidates = sorted(path.rglob('*')) if path.is_dir() else [path]
        found += [item for item in candidates if item.suffix in {'.py', '.md'} and '__pycache__' not in item.parts]
    return found


def load_moves(paths: Iterable[Path]) -> list[Move]:
    """Return the moves of tables, reading every TOML and JSON file of a directory in name order.

    Args:
        paths:
            The table files, or directories of table files.

    Returns:
        moves:
            The moves of every table, in order.

    Raises:
        ValueError:
            A move misses its old or its new module.
    """
    moves = []
    for path in paths:
        tables = sorted([*path.glob('*.toml'), *path.glob('*.json')]) if path.is_dir() else [path]
        for table in tables:
            content = json.loads(table.read_text()) if table.suffix == '.json' else tomllib.loads(table.read_text())
            for item in content.get('moves', []):
                if not {'old', 'new'} <= set(item):
                    error_msg = f'A move of {table} misses its old or its new module: {item}.'
                    raise ValueError(error_msg)
                names = frozenset(item['names']) if 'names' in item else None
                moves.append(Move(item['old'], item['new'], names))
    return moves


def rewrite_source(source: str, moves: Sequence[Move], *, package: str | None = None) -> str:
    """Return Python source whose import statements and doctest imports follow the moves.

    Args:
        source:
            The Python source.

        moves:
            The moves.

        package:
            The dotted package of the source, which relative imports resolve against, or `None`.

    Returns:
        source:
            The rewritten source, unformatted.
    """
    return _rewrite_doctests(_rewrite_statements(source, moves, package), moves)


def rewrite_file(path: Path, moves: Sequence[Move], mode: black.Mode) -> bool:
    """Rewrite the imports of one Python or Markdown file in place, and return whether it changed.

    Args:
        path:
            The file.

        moves:
            The moves.

        mode:
            The black mode Python files are formatted with.

    Returns:
        changed:
            Whether the file changed.
    """
    text = path.read_text()
    if path.suffix == '.md':
        rewritten = _rewrite_markdown(text, moves)
    else:
        rewritten = rewrite_source(text, moves, package=_package_of(path))
        if rewritten != text:
            rewritten = black.format_str(rewritten, mode=mode)
    if rewritten == text:
        return False
    path.write_text(rewritten)
    return True


def main(arguments: Sequence[str] | None = None) -> None:
    """Rewrite the imports of the files under the given paths from the given tables."""
    parser = argparse.ArgumentParser(description='Rewrite imports after names or modules move.')
    parser.add_argument('--table', action='append', type=Path, required=True, help='a table, or a directory of them')
    parser.add_argument('--pyproject', type=Path, default=Path('pyproject.toml'), help='the black configuration')
    parser.add_argument('paths', nargs='+', type=Path, help='files or directories to rewrite')
    options = parser.parse_args(arguments)
    moves = load_moves(options.table)
    mode = _black_mode(options.pyproject)
    changed = [path for path in _files(options.paths) if rewrite_file(path, moves, mode)]
    for path in changed:
        print(f'rewrote {path}')  # noqa: T201


if __name__ == '__main__':
    main()
