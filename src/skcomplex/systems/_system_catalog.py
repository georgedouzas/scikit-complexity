"""Define the catalog of systems, which lists them from their sources and imports a system's module when used."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import ast
import dataclasses
import difflib
import importlib
import inspect
from pathlib import Path

from ..core.errors import RegistryError
from ._system_entry import SystemEntry
from ._system_listing import SystemListing

LISTED = ('name', 'field', 'description', 'tags', 'references')


class SystemCatalog:
    """Catalog of the systems of a package: their listings read from source, and their entries once registered.

    The catalog lists every factory that a module of the package decorates with `register_system`, reading the
    registrations from the sources without importing them. Asking for the entry of a listed system imports its module,
    whose registration adds the entry. Systems registered at run time outside the package are listed too.

    Args:
        root:
            The directory of the system package, such as the directory of `skcomplex.systems`.

        package:
            The dotted name of the system package.

    Examples:
        >>> from importlib.resources import files
        >>> from pathlib import Path
        >>> from skcomplex.systems import SystemCatalog
        >>> catalog = SystemCatalog(Path(str(files('skcomplex.systems'))))
        >>> catalog.listings(field='biology/ecology')[0].field
        'biology/ecology'
    """

    def __init__(self, root: Path, package: str = 'skcomplex.systems') -> None:
        self.root = root
        self.package = package
        self._entries: dict[str, SystemEntry] = {}
        self._scanned: dict[str, SystemListing] | None = None

    def register(self, entry: SystemEntry) -> None:
        """Register a system under its name, replacing any previous one.

        An entry without a description takes the first line of its factory's docstring, without a leading `Return`.

        Args:
            entry:
                The registered system.
        """
        if not entry.description:
            entry = dataclasses.replace(entry, description=self._summarize(inspect.getdoc(entry.factory) or ''))
        self._entries[entry.name] = entry

    def entry(self, name: str) -> SystemEntry:
        """Return a system by name, importing the module of a listed system so that it registers.

        Args:
            name:
                The name of the system.

        Returns:
            entry:
                The registered system.

        Raises:
            RegistryError:
                No system has the name, with the closest names in the message.
        """
        if name not in self._entries:
            listing = self._listed().get(name)
            if listing is not None:
                importlib.import_module(listing.module)
        if name not in self._entries:
            known = sorted({*self._listed(), *self._entries})
            close = difflib.get_close_matches(name, known, n=3)
            hint = f' Did you mean {", ".join(close)}?' if close else ' List the systems with `catalog()`.'
            error_msg = f'No system is registered as {name!r}.{hint}'
            raise RegistryError(error_msg)
        return self._entries[name]

    def listings(self, field: str | None = None, tag: str | None = None) -> list[SystemListing]:
        """Return the listed systems, optionally of one field or with one tag, sorted by field and name.

        Args:
            field:
                The field, such as `biology`, or the field path, such as `biology/ecology`, to keep, or `None` for
                every field.

            tag:
                The tag to keep, or `None` for every system.

        Returns:
            listings:
                The listed systems.

        Raises:
            RegistryError:
                A system registers its name, field, description, tags, or references with a value computed at run time.
        """
        listings = dict(self._listed())
        for name, entry in self._entries.items():
            listings.setdefault(
                name,
                SystemListing(
                    entry.name,
                    entry.field,
                    entry.description,
                    entry.tags,
                    entry.references,
                    entry.factory.__module__,
                ),
            )
        return sorted(
            (
                listing
                for listing in listings.values()
                if (field is None or listing.field == field or listing.field.startswith(f'{field}/'))
                and (tag is None or tag in listing.tags)
            ),
            key=lambda listing: (listing.field, listing.name),
        )

    def _listed(self) -> dict[str, SystemListing]:
        """Return the listing of every system of the package, by name, read once from its sources."""
        if self._scanned is None:
            found: dict[str, SystemListing] = {}
            for path in sorted(self.root.rglob('_*.py')):
                if path.name == '__init__.py' or path.parent == self.root:
                    continue
                module = '.'.join((self.package, *path.relative_to(self.root).with_suffix('').parts))
                tree = ast.parse(path.read_text())
                calls = [(node, self._registration(node)) for node in tree.body if isinstance(node, ast.FunctionDef)]
                constants = self._constants(path) if any(call is not None for _, call in calls) else {}
                for node, call in calls:
                    if call is not None:
                        listing = self._listing(node, call, module, constants)
                        found[listing.name] = listing
            self._scanned = found
        return self._scanned

    def _listing(
        self,
        node: ast.FunctionDef,
        call: ast.Call,
        module: str,
        constants: dict[str, object],
    ) -> SystemListing:
        """Return the listing of a factory from its registration and its docstring."""
        values = {keyword: self._literal(call, keyword, module, constants) for keyword in LISTED}
        field = values['field']
        if not isinstance(field, str):
            error_msg = f'The system {node.name} of {module} registers no field path. Give `field` as a string.'
            raise RegistryError(error_msg)
        return SystemListing(
            name=str(values['name'] or node.name),
            field=field,
            description=str(values['description'] or self._summarize(ast.get_docstring(node) or '')),
            tags=self._strings(values['tags'], 'tags', module),
            references=self._strings(values['references'], 'references', module),
            module=module,
        )

    @staticmethod
    def _strings(value: object, keyword: str, module: str) -> tuple[str, ...]:
        """Return the strings of a registered tuple or list, empty when the registration does not give it."""
        if value is None:
            return ()
        if not isinstance(value, tuple | list):
            error_msg = (
                f'A system of {module} registers the argument {keyword!r} as {value!r}. Give a tuple of strings.'
            )
            raise RegistryError(error_msg)
        return tuple(str(item) for item in value)

    def _literal(self, call: ast.Call, keyword: str, module: str, constants: dict[str, object]) -> object:
        """Return the static value of a keyword argument of a registration, or `None` when it is not given."""
        node = next((item.value for item in call.keywords if item.arg == keyword), None)
        if node is None and keyword == 'name' and call.args:
            node = call.args[0]
        if node is None:
            return None
        try:
            return self._evaluate(node, constants)
        except (TypeError, ValueError) as error:
            error_msg = (
                f'A system of {module} registers the argument {keyword!r} with a value computed at run time. Write it '
                'as a literal or a module-level literal constant, so the catalog lists the system without importing '
                'its module.'
            )
            raise RegistryError(error_msg) from error

    def _constants(self, path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, object]:
        """Return the module-level literal constants of a system module, with those it imports from its own package."""
        if path in seen or not path.exists():
            return {}
        constants: dict[str, object] = {}
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                source = self._constants(path.with_name(f'{node.module}.py'), seen | {path})
                constants.update(
                    {alias.asname or alias.name: source[alias.name] for alias in node.names if alias.name in source},
                )
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    constants[node.targets[0].id] = self._evaluate(node.value, constants)
                except (TypeError, ValueError):
                    continue
        return constants

    def _evaluate(self, node: ast.expr, constants: dict[str, object]) -> object:
        """Return the value of a literal expression whose names are module-level literal constants."""
        if isinstance(node, ast.Name):
            if node.id in constants:
                return constants[node.id]
            error_msg = f'The name {node.id} is not a literal constant of the module.'
            raise ValueError(error_msg)
        if isinstance(node, ast.Tuple | ast.List):
            items: list[object] = []
            for element in node.elts:
                if isinstance(element, ast.Starred):
                    unpacked = self._evaluate(element.value, constants)
                    if not isinstance(unpacked, tuple | list):
                        error_msg = f'The unpacked value {unpacked!r} is not a tuple or a list.'
                        raise TypeError(error_msg)
                    items.extend(unpacked)
                else:
                    items.append(self._evaluate(element, constants))
            return tuple(items) if isinstance(node, ast.Tuple) else items
        return ast.literal_eval(node)

    @staticmethod
    def _registration(node: ast.FunctionDef) -> ast.Call | None:
        """Return the `register_system` call that decorates a factory, or `None` for an undecorated function."""
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            target = decorator.func
            name = target.id if isinstance(target, ast.Name) else getattr(target, 'attr', None)
            if name == 'register_system':
                return decorator
        return None

    @staticmethod
    def _summarize(docstring: str) -> str:
        r"""Return the description of a system from its factory's docstring: the first line, without a leading `Return`.

        Args:
            docstring:
                The docstring of the factory.

        Returns:
            summary:
                The first line of the docstring, without a leading `Return`.

        Examples:
            >>> from skcomplex.systems import SystemCatalog
            >>> SystemCatalog._summarize('Return the Lotka-Volterra predator and prey.\n\nMore text.')
            'The Lotka-Volterra predator and prey.'
        """
        summary = docstring.split('\n', maxsplit=1)[0]
        if summary.startswith('Return '):
            summary = summary.removeprefix('Return ')
            summary = summary[0].upper() + summary[1:]
        return summary
