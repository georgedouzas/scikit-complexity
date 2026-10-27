"""List the models of the library from their registrations, and import a model's module only when it is used."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import ast
import difflib
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from ..core.errors import RegistryError
from ..core.system import System
from ..engine.runtime import MODELS, ModelEntry, Simulation, summarize
from ..engine.runtime import create as create_registered
from ..engine.runtime import describe as describe_registered
from ..engine.runtime import run as run_registered

PACKAGE = Path(str(files('skcomplex.models')))
LISTED = ('name', 'field', 'description', 'tags', 'references')


def _registration(node: ast.FunctionDef) -> ast.Call | None:
    """Return the `register_model` call that decorates a factory, or `None` for an undecorated function."""
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        target = decorator.func
        name = target.id if isinstance(target, ast.Name) else getattr(target, 'attr', None)
        if name == 'register_model':
            return decorator
    return None


def _constants(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Return the module-level literal constants of a model module, with those it imports from its own package."""
    if path in seen or not path.exists():
        return {}
    constants: dict[str, Any] = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            source = _constants(path.with_name(f'{node.module}.py'), seen | {path})
            constants.update(
                {alias.asname or alias.name: source[alias.name] for alias in node.names if alias.name in source},
            )
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = _evaluate(node.value, constants)
            except ValueError:
                continue
    return constants


def _evaluate(node: ast.expr, constants: dict[str, Any]) -> Any:  # noqa: ANN401
    """Return the value of a literal expression whose names are module-level literal constants."""
    if isinstance(node, ast.Name):
        if node.id in constants:
            return constants[node.id]
        error_msg = f'The name {node.id} is not a literal constant of the module.'
        raise ValueError(error_msg)
    if isinstance(node, ast.Tuple | ast.List):
        items: list[Any] = []
        for element in node.elts:
            if isinstance(element, ast.Starred):
                items.extend(_evaluate(element.value, constants))
            else:
                items.append(_evaluate(element, constants))
        return tuple(items) if isinstance(node, ast.Tuple) else items
    return ast.literal_eval(node)


def _literal(call: ast.Call, keyword: str, module: str, constants: dict[str, Any]) -> Any:  # noqa: ANN401
    """Return the static value of a keyword argument of a registration, or `None` when it is not given."""
    node = next((item.value for item in call.keywords if item.arg == keyword), None)
    if node is None and keyword == 'name' and call.args:
        node = call.args[0]
    if node is None:
        return None
    try:
        return _evaluate(node, constants)
    except ValueError as error:
        error_msg = (
            f'A model of {module} registers the argument {keyword!r} with a value computed at run time. Write it as '
            'a literal or a module-level literal constant, so the catalog lists the model without importing its module.'
        )
        raise RegistryError(error_msg) from error


def _listing(node: ast.FunctionDef, call: ast.Call, module: str, constants: dict[str, Any]) -> 'ModelListing':
    """Return the listing of a factory from its registration and its docstring."""
    values = {keyword: _literal(call, keyword, module, constants) for keyword in LISTED}
    field = values['field']
    if not isinstance(field, str):
        error_msg = f'The model {node.name} of {module} registers no field path. Give `field` as a string.'
        raise RegistryError(error_msg)
    return ModelListing(
        name=values['name'] or node.name,
        field=field,
        description=values['description'] or summarize(ast.get_docstring(node) or ''),
        tags=tuple(values['tags'] or ()),
        references=tuple(values['references'] or ()),
        module=module,
    )


@cache
def _listings() -> dict[str, 'ModelListing']:
    """Return the listing of every model of the library, by name, read once per process."""
    return scan_listings(PACKAGE)


def _registered_listing(entry: ModelEntry) -> 'ModelListing':
    """Return the listing of a model registered at run time, outside the modules of the library."""
    return ModelListing(
        entry.name,
        entry.field,
        entry.description,
        entry.tags,
        entry.references,
        entry.factory.__module__,
    )


def _load(key: object) -> None:
    """Import the module of a listed model, so that resolving its name in the model registry finds it."""
    listing = _listings().get(key) if isinstance(key, str) else None
    if listing is not None:
        importlib.import_module(listing.module)


def _imported(name: str) -> str:
    """Import the module of a listed model, so it registers, and return the name; raise an error for unknown names."""
    listing = _listings().get(name)
    if listing is not None:
        importlib.import_module(listing.module)
        return name
    if name in MODELS.keys:
        return name
    known = sorted({*_listings(), *map(str, MODELS.keys)})
    close = difflib.get_close_matches(name, known, n=3)
    hint = f' Did you mean {", ".join(close)}?' if close else ' List the models with `catalog()`.'
    error_msg = f'No model is registered as {name!r}.{hint}'
    raise RegistryError(error_msg)


def scan_listings(root: Path) -> dict[str, 'ModelListing']:
    """Return the listing of every model a package registers, by name, read from its sources without importing them.

    Args:
        root:
            The directory of a model package, such as the directory of `skcomplex.models`.

    Returns:
        listings:
            The listing of every factory decorated with `register_model`, by model name.

    Raises:
        RegistryError:
            A model registers its name, field, description, tags, or references with a value computed at run time.

    Examples:
        >>> from importlib.resources import files
        >>> from pathlib import Path
        >>> from skcomplex.models import scan_listings
        >>> listings = scan_listings(Path(str(files('skcomplex.models'))))
        >>> listings['lotka_volterra'].module
        'skcomplex.models.biology.ecology._factories'
    """
    found: dict[str, ModelListing] = {}
    for path in sorted(root.rglob('_*.py')):
        if path.name == '__init__.py' or path.parent == root:
            continue
        module = '.'.join(('skcomplex', 'models', *path.relative_to(root).with_suffix('').parts))
        tree = ast.parse(path.read_text())
        calls = [(node, _registration(node)) for node in tree.body if isinstance(node, ast.FunctionDef)]
        constants = _constants(path) if any(call is not None for _, call in calls) else {}
        for node, call in calls:
            if call is not None:
                listing = _listing(node, call, module, constants)
                found[listing.name] = listing
    return found


def catalog(field: str | None = None, tag: str | None = None) -> list['ModelListing']:
    """Return the listed models, optionally of one field or with one tag, sorted by field and name.

    The listing reads the registration of every model of the library from its source, and imports no model module.
    Models registered at run time outside the library are listed too.

    Args:
        field:
            The field, such as `biology`, or the field path, such as `biology/ecology`, to keep, or `None` for every
            field.

        tag:
            The tag to keep, or `None` for every model.

    Returns:
        listings:
            The listed models.

    Raises:
        RegistryError:
            A model registers its name, field, description, tags, or references with a value computed at run time.

    Examples:
        >>> from skcomplex.models import catalog
        >>> 'lotka_volterra' in [listing.name for listing in catalog(field='biology/ecology')]
        True
        >>> sorted({listing.field for listing in catalog(field='chemistry')})
        ['chemistry/kinetics', 'chemistry/pattern_formation']
    """
    listings = dict(_listings())
    for name in MODELS.keys:
        listings.setdefault(str(name), _registered_listing(MODELS.resolve(name)))
    return sorted(
        (
            listing
            for listing in listings.values()
            if (field is None or listing.field == field or listing.field.startswith(f'{field}/'))
            and (tag is None or tag in listing.tags)
        ),
        key=lambda listing: (listing.field, listing.name),
    )


def describe(name: str) -> ModelEntry:
    """Return a model by name, with its factory, parameters, metadata, and default run settings.

    It imports the model's module and nothing else of the library.

    Args:
        name:
            The name of the model.

    Returns:
        entry:
            The registered model.

    Raises:
        RegistryError:
            No model has the name.

    Examples:
        >>> from skcomplex.models import describe
        >>> describe('lotka_volterra').field
        'biology/ecology'
    """
    return describe_registered(_imported(name))


def create(name: str, **parameters: Any) -> System:  # noqa: ANN401
    """Return the system of a model, built by its factory.

    It imports the model's module and nothing else of the library.

    Args:
        name:
            The name of the model.

        **parameters:
            Arguments of the factory that replace its defaults.

    Returns:
        system:
            The model's system.

    Raises:
        RegistryError:
            No model has the name.

    Examples:
        >>> from skcomplex.models import create
        >>> len(create('lotka_volterra').components)
        2
    """
    return create_registered(_imported(name), **parameters)


def run(name: str, *, settings: Mapping[str, Any] | None = None, **parameters: Any) -> Simulation:  # noqa: ANN401
    """Create a model and simulate it with its default run settings.

    It imports the model's module and nothing else of the library.

    Args:
        name:
            The name of the model.

        settings:
            Run settings that replace the model's defaults: `until` and any `Simulation` argument.

        **parameters:
            Arguments of the factory that replace its defaults.

    Returns:
        simulation:
            The finished simulation, whose `trajectory_` holds the run.

    Raises:
        RegistryError:
            No model has the name.
    """
    return run_registered(_imported(name), settings=settings, **parameters)


@dataclass(frozen=True)
class ModelListing:
    """Model as the catalog lists it: its name and metadata, and the module that registers it.

    Args:
        name:
            The name the model is created by.

        field:
            The field path of the model, such as `biology/ecology`.

        description:
            One sentence on what the model shows.

        tags:
            Keywords for filtering.

        references:
            The sources of the model.

        module:
            The dotted name of the module that registers the model.
    """

    name: str
    field: str
    description: str
    tags: tuple[str, ...]
    references: tuple[str, ...]
    module: str


MODELS.register_loader(_load)
