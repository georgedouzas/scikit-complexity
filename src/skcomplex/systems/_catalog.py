"""Find the system classes of the library by field, topics, and capability tags."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import ast
import dataclasses
import importlib
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path

from ..core.components.systems import BaseSystem
from ..core.errors import RegistryError, ValidationError
from ..core.specs import Tags

_LISTED = ('field', 'topics', 'references')


class _SystemClasses:
    """Reader of the system classes of a package.

    It finds them in the sources without importing them. A system class is a class whose body gives the class variable
    `field`. Its field and topics are read from the sources, so selecting classes by them imports only the modules of
    the selected classes.

    Args:
        root:
            The directory of the system package.

        package:
            The dotted name of the system package.
    """

    def __init__(self, root: Path, package: str = 'skcomplex.systems') -> None:
        self.root = root
        self.package = package
        self._found: dict[str, tuple[str, str, tuple[str, ...]]] | None = None

    def classes(
        self,
        field: str | None = None,
        topics: str | Sequence[str] = (),
        tags: Mapping[str, str | Sequence[str]] | None = None,
    ) -> list[tuple[str, type[BaseSystem]]]:
        """Return the name and the class of every system class.

        The classes can be selected by field, by topics, or by tags.

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
            RegistryError:
                A system class is not named with the suffix `System`, or gives its field, topics, or references with a
                value computed at run time.

            ValidationError:
                A tag names no field of `Tags`.
        """
        wanted = {topics} if isinstance(topics, str) else set(topics)
        found = []
        for name, (module, path, listed) in self._get_listed().items():
            if field is not None and path != field and not path.startswith(f'{field}/'):
                continue
            if not wanted <= set(listed):
                continue
            system_class: type[BaseSystem] = getattr(importlib.import_module(module), name)
            if tags is None or self._has_tags(system_class().get_tags(), tags):
                found.append((name, system_class))
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

    def _get_listed(self) -> dict[str, tuple[str, str, tuple[str, ...]]]:
        """Return the module, the field path, and the topics of every system class.

        They are given by class name and read once.
        """
        if self._found is None:
            found: dict[str, tuple[str, str, tuple[str, ...]]] = {}
            for path in sorted(self.root.rglob('_*.py')):
                if path.name == '__init__.py' or path.parent == self.root:
                    continue
                module = '.'.join((self.package, *path.relative_to(self.root).with_suffix('').parts))
                definitions = [
                    node
                    for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.ClassDef) and self._class_value(node, 'field') is not None
                ]
                constants = self._constants(path) if definitions else {}
                for node in definitions:
                    found[node.name] = self._class_listing(node, module, constants)
            self._found = found
        return self._found

    def _class_listing(
        self,
        node: ast.ClassDef,
        module: str,
        constants: dict[str, object],
    ) -> tuple[str, str, tuple[str, ...]]:
        """Return the module, the field path, and the topics of a system class.

        They are read from its class variables.

        Args:
            node:
                The definition of the class.

            module:
                The dotted name of the module.

            constants:
                The module-level literal constants of the module.

        Returns:
            listing:
                The module, the field path, and the topics.

        Raises:
            RegistryError:
                The class is not named with the suffix `System`, gives its field path as no string, or gives a listed
                class variable with a value computed at run time.
        """
        values = {name: self._literal(self._class_value(node, name), name, module, constants) for name in _LISTED}
        field = values['field']
        if not node.name.endswith('System') or not isinstance(field, str):
            error_msg = (
                f'The class {node.name} of {module} gives the class variable `field`, so it is a system class. Name it '
                'with the suffix `System`, and give `field` as a string.'
            )
            raise RegistryError(error_msg)
        for name in ('topics', 'references'):
            if values[name] is not None and not isinstance(values[name], tuple | list):
                error_msg = f'The class {node.name} of {module} gives {name!r} as {values[name]!r}. Give a tuple.'
                raise RegistryError(error_msg)
        topics = values['topics']
        return module, field, tuple(str(topic) for topic in topics) if isinstance(topics, tuple | list) else ()

    @staticmethod
    def _class_value(node: ast.ClassDef, name: str) -> ast.expr | None:
        """Return the value a class body assigns to a class variable.

        It is `None` when the body assigns none.
        """
        for item in node.body:
            if isinstance(item, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in item.targets
            ):
                return item.value
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.target.id == name:
                return item.value
        return None

    def _literal(self, node: ast.expr | None, name: str, module: str, constants: dict[str, object]) -> object:
        """Return the static value of a class variable.

        It is `None` when the class does not give it.

        Args:
            node:
                The expression of the value, or `None` when it is not given.

            name:
                The name of the class variable, for the message of the error.

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
                f'A system of {module} defines the class variable {name!r} with a value computed at run time. Write it '
                'as a literal or a module-level literal constant, so `all_systems` reads it without importing the '
                'module.'
            )
            raise RegistryError(error_msg) from error

    def _constants(self, path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, object]:
        """Return the module-level literal constants of a system module.

        They include those it imports from its own package.
        """
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
        """Return the value of a literal expression.

        Its names are module-level literal constants.
        """
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


_SYSTEM_CLASSES = _SystemClasses(Path(str(files('skcomplex.systems'))))


def all_systems(
    field: str | None = None,
    topics: str | Sequence[str] = (),
    tags: Mapping[str, str | Sequence[str]] | None = None,
) -> list[tuple[str, type[BaseSystem]]]:
    """Return the name and the class of every ready-made system.

    The systems can be selected by field, by topics, or by capability tags. It takes the role of
    `sklearn.utils.all_estimators`. The field and the topics of every class are read from the sources, so selecting by
    them imports the modules of the selected classes alone. Filtering by capability tags builds every selected class
    with its default model parameters and reads its tags.

    Args:
        field:
            The field, such as `chemistry`, or the field path, such as `chemistry/kinetics`, to keep, or `None` for
            every field.

        topics:
            The topic, or the topics, that every kept system lists, such as `'oscillation'`.

        tags:
            The capability tags to keep, by field of `Tags`, each a value or a sequence of values that the tags of a
            kept system hold, such as `{'randomness': 'stochastic'}`, or `None` for any tags.

    Returns:
        systems:
            The name and the class of every kept system, sorted by name.

    Raises:
        RegistryError:
            A system class is not named with the suffix `System`, or gives its field, topics, or references with a
            value computed at run time.

        ValidationError:
            A tag names no field of `Tags`.

    Examples:
        >>> from skcomplex.systems import all_systems
        >>> [name for name, _ in all_systems(field='chemistry/pattern_formation')]
        ['BrusselatorSystem', 'GrayScottSystem', 'SchnakenbergSystem']
        >>> [name for name, _ in all_systems(field='chemistry', tags={'randomness': 'stochastic'})]
        ['ChemicalLangevinEquationSystem', 'ChemicalLangevinNetworkSystem']
    """
    return _SYSTEM_CLASSES.classes(field, topics, tags)
