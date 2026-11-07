"""Define the catalog of systems, which lists them from their sources and imports a system's module when used."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import ast
import dataclasses
import difflib
import functools
import importlib
import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from ..core.components.system import BaseSystem, System
from ..core.errors import RegistryError, ValidationError
from ..core.specs import ObservableSpec, Tags
from ._system_entry import SystemEntry
from ._system_listing import SystemListing
from .types import Factory

LISTED = ('name', 'field', 'description', 'tags', 'references')
CLASS_LISTED = ('field', 'topics', 'references')
TUNED = ('until', 'dt', 'solver', 'initial_state')


class SystemCatalog:
    """Catalog of the systems of a package: their listings read from source, and their entries once registered.

    The catalog lists every factory that a module of the package decorates with `register_system`, and every system
    class of a module, a class whose body gives the class variable `field`, reading both from the sources without
    importing them. Asking for the entry of a listed factory imports its module, whose registration adds the entry.
    Asking for the entry of a listed class imports its module and adds an entry whose factory builds the composite of
    the class for its model parameters, so code that creates systems by name runs system classes unchanged. Systems
    registered at run time outside the package are listed too.

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
            if listing is not None and listing.class_name is not None:
                self.register(self._class_entry(listing, self._import_class(listing)))
            elif listing is not None:
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

    def classes(
        self,
        field: str | None = None,
        topics: str | Sequence[str] = (),
        tags: Mapping[str, str | Sequence[str]] | None = None,
    ) -> list[tuple[str, type[BaseSystem]]]:
        """Return the name and the class of every system class, optionally of a field, with topics, or with tags.

        The field and the topics are read from the sources, so only the modules of the selected classes are imported.
        Filtering by capability tags builds every selected class with its default model parameters and reads its tags.

        Args:
            field:
                The field, such as `chemistry`, or the field path, such as `chemistry/kinetics`, to keep, or `None` for
                every field.

            topics:
                The topic, or the topics, that every kept class lists.

            tags:
                The capability tags to keep, by field of `Tags`, each a value or a sequence of values that the tags of a
                kept class hold, or `None` for any tags.

        Returns:
            classes:
                The name and the class of every kept system class, sorted by name.

        Raises:
            ValidationError:
                A tag names no field of `Tags`.
        """
        wanted = {topics} if isinstance(topics, str) else set(topics)
        found = []
        for listing in self.listings(field):
            if listing.class_name is None or not wanted <= set(listing.tags):
                continue
            system_class = self._import_class(listing)
            if tags is None or self._has_tags(system_class().get_tags(), tags):
                found.append((listing.class_name, system_class))
        return sorted(found, key=lambda pair: pair[0])

    @staticmethod
    def _has_tags(tags: Tags, wanted: Mapping[str, str | Sequence[str]]) -> bool:
        """Return whether capability tags hold every wanted value of every field.

        Args:
            tags:
                The tags of a system.

            wanted:
                The wanted values, by field of `Tags`, each a value or a sequence of values.

        Returns:
            held:
                Whether the tags hold every wanted value.

        Raises:
            ValidationError:
                A wanted field is no field of `Tags`.
        """
        names = [item.name for item in dataclasses.fields(Tags)]
        for name, value in wanted.items():
            if name not in names:
                error_msg = f'The tag {name!r} is unknown. Use one of {", ".join(map(repr, names))}.'
                raise ValidationError(error_msg)
            actual = getattr(tags, name)
            held = {actual} if isinstance(actual, str) or actual is None else set(actual)
            if not ({value} if isinstance(value, str) else set(value)) <= held:
                return False
        return True

    @staticmethod
    def _import_class(listing: SystemListing) -> type[BaseSystem]:
        """Return the system class of a listing, importing its module."""
        system_class: type[BaseSystem] = getattr(importlib.import_module(listing.module), str(listing.class_name))
        return system_class

    @staticmethod
    def _class_entry(listing: SystemListing, system_class: type[BaseSystem]) -> SystemEntry:
        """Return the entry of a system class: a factory of its composite, its metadata, and its tuned run settings.

        Args:
            listing:
                The listing of the class.

            system_class:
                The class.

        Returns:
            entry:
                The entry, whose factory takes the model parameters of the class and returns the prepared composite of
                its parts, and whose default run settings are the tuned run settings of the class.
        """
        build: Callable[..., BaseSystem] = system_class

        @functools.wraps(system_class, updated=())
        def factory(**parameters: object) -> System:
            return build(**parameters).prepare().system_

        def observables(system: System) -> Sequence[ObservableSpec]:
            return tuple(system_class.observables)

        defaults = {name: getattr(system_class, name) for name in TUNED if getattr(system_class, name) is not None}
        created: Factory = factory
        return SystemEntry(
            listing.name,
            created,
            listing.field,
            listing.description,
            listing.tags,
            listing.references,
            defaults,
            observables,
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
                classes = [
                    node
                    for node in tree.body
                    if isinstance(node, ast.ClassDef) and self._class_value(node, 'field') is not None
                ]
                constants = self._constants(path) if classes or any(call is not None for _, call in calls) else {}
                for node, call in calls:
                    if call is not None:
                        listing = self._listing(node, call, module, constants)
                        found[listing.name] = listing
                for definition in classes:
                    listing = self._class_listing(definition, module, constants)
                    found[listing.name] = listing
            self._scanned = found
        return self._scanned

    def _class_listing(self, node: ast.ClassDef, module: str, constants: dict[str, object]) -> SystemListing:
        """Return the listing of a system class from its class variables and its docstring.

        Args:
            node:
                The definition of the class.

            module:
                The dotted name of the module.

            constants:
                The module-level literal constants of the module.

        Returns:
            listing:
                The listing, under the name of the class in snake case, with the topics as its tags.

        Raises:
            RegistryError:
                The class is not named with the suffix `System`, or gives its field path as no string.
        """
        values = {
            name: self._literal(
                self._class_value(node, name),
                f'defines the class variable {name!r}',
                module,
                constants,
            )
            for name in CLASS_LISTED
        }
        field = values['field']
        if not node.name.endswith('System') or not isinstance(field, str):
            error_msg = (
                f'The class {node.name} of {module} gives the class variable `field`, so it is a system class. Name it '
                'with the suffix `System`, and give `field` as a string.'
            )
            raise RegistryError(error_msg)
        return SystemListing(
            name=self._snake_case(node.name),
            field=field,
            description=self._summarize(ast.get_docstring(node) or ''),
            tags=self._strings(values['topics'], 'topics', module),
            references=self._strings(values['references'], 'references', module),
            module=module,
            class_name=node.name,
        )

    @staticmethod
    def _class_value(node: ast.ClassDef, name: str) -> ast.expr | None:
        """Return the value a class body assigns to a class variable, or `None` when it assigns none."""
        for item in node.body:
            if isinstance(item, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in item.targets
            ):
                return item.value
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.target.id == name:
                return item.value
        return None

    @staticmethod
    def _snake_case(class_name: str) -> str:
        """Return the catalog name of a system class: its name without the suffix `System`, in snake case.

        Args:
            class_name:
                The name of the class.

        Returns:
            name:
                The name in snake case, with an acronym kept as one word.

        Examples:
            >>> from skcomplex.systems import SystemCatalog
            >>> SystemCatalog._snake_case('GrayScottSystem'), SystemCatalog._snake_case('BrusselatorODESystem')
            ('gray_scott', 'brusselator_ode')
        """
        stem = class_name.removesuffix('System')
        return re.sub(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', '_', stem).lower()

    def _listing(
        self,
        node: ast.FunctionDef,
        call: ast.Call,
        module: str,
        constants: dict[str, object],
    ) -> SystemListing:
        """Return the listing of a factory from its registration and its docstring."""
        values = {
            keyword: self._literal(
                self._keyword_value(call, keyword),
                f'registers the argument {keyword!r}',
                module,
                constants,
            )
            for keyword in LISTED
        }
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
            error_msg = f'A system of {module} gives {keyword!r} as {value!r}. Give a tuple of strings.'
            raise RegistryError(error_msg)
        return tuple(str(item) for item in value)

    @staticmethod
    def _keyword_value(call: ast.Call, keyword: str) -> ast.expr | None:
        """Return the value of a keyword argument of a registration, or `None` when it is not given."""
        node = next((item.value for item in call.keywords if item.arg == keyword), None)
        if node is None and keyword == 'name' and call.args:
            node = call.args[0]
        return node

    def _literal(self, node: ast.expr | None, given: str, module: str, constants: dict[str, object]) -> object:
        """Return the static value of a registered argument or a class variable, or `None` when it is not given.

        Args:
            node:
                The expression of the value, or `None` when it is not given.

            given:
                What gives the value, such as `registers the argument 'field'`, for the message of the error.

            module:
                The dotted name of the module.

            constants:
                The module-level literal constants of the module.

        Returns:
            value:
                The value, or `None`.

        Raises:
            RegistryError:
                The value is computed at run time.
        """
        if node is None:
            return None
        try:
            return self._evaluate(node, constants)
        except (TypeError, ValueError) as error:
            error_msg = (
                f'A system of {module} {given} with a value computed at run time. Write it as a literal or a '
                'module-level literal constant, so the catalog lists the system without importing its module.'
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
