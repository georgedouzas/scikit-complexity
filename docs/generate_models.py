"""Generate one page per model with its pattern, one index per field, and the navigation of the models."""

import ast
import inspect
import sys
import tomllib
from importlib.resources import files
from pathlib import Path
from typing import Any

import mkdocs_gen_files
from gallery_tools import Example, build_from_environment
from inventory_checks import InventoryError, inventory_problems

from skcomplex.analysis import SymbolicAnalysis
from skcomplex.core.system import System
from skcomplex.errors import AnalysisError
from skcomplex.models import catalog, create, describe

DOCS = Path(str(files('gallery_tools'))).parent
INVENTORY = DOCS / 'overview' / 'inventory.toml'
GALLERY = build_from_environment(DOCS)
EQUATION_FORMS = {
    'ordinary differential equations',
    'fractional differential equations',
    'stochastic differential equations',
    'delay differential equations',
    'discrete map',
}


def _inventory() -> dict[str, Any]:
    """Return the inventory by field and subfield."""
    data = tomllib.loads(INVENTORY.read_text())
    for section in ('sources', 'primitives', 'vocabulary'):
        data.pop(section)
    return data


def _titles(inventory: dict[str, Any]) -> dict[str, str]:
    """Return the title of every model of the inventory, by name."""
    return {
        name: entry['title']
        for subfields in inventory.values()
        for models in subfields.values()
        for name, entry in models.items()
    }


def _title(name: str) -> str:
    """Return the title of a field or a subfield."""
    return name.replace('_', ' ').capitalize()


def _examples() -> dict[str, Example]:
    """Return the gallery example of every model the build selects."""
    return {example.model: example for example in GALLERY.selected()}


def _page_of(example: Example) -> str:
    """Return the path of the gallery page of an example, relative to the documentation folder."""
    return (GALLERY.generated_dir(example) / f'{example.path.stem}.md').relative_to(DOCS).as_posix()


def _gallery_lines(name: str, examples: dict[str, Example]) -> list[str]:
    """Return the screenshot of a model and the link to its example, for a page three folders below the docs."""
    example = examples.get(name)
    if example is None:
        return []
    page = f'../../../{_page_of(example)}'
    screenshot = GALLERY.screenshot(example)
    lines = [f'[Example]({page})', '']
    if screenshot.exists():
        image = f'../../../{screenshot.relative_to(DOCS).as_posix()}'
        lines = [f'[![The {name} example]({image}){{ width="480" }}]({page})', '', *lines]
    return lines


def _component_types(system: System) -> list[type]:
    """Return the element and interaction types of a system and its subsystems, in order of first use."""
    types: list[type] = []
    for component in [*system.elements, *system.interactions]:
        kind = getattr(component, 'element', None) or type(component)
        if kind not in types:
            types.append(kind)
    for subsystem in system.systems:
        types += [kind for kind in _component_types(subsystem) if kind not in types]
    return types


def _class_source(kind: type) -> str:
    """Return the source of an element or interaction type, its class statement or the assignment that builds it."""
    try:
        return inspect.getsource(kind)
    except (OSError, TypeError):
        pass
    module = sys.modules[kind.__module__]
    text = inspect.getsource(module)
    for node in ast.parse(text).body:
        targets = [getattr(target, 'id', None) for target in getattr(node, 'targets', [])]
        if kind.__name__ in targets:
            return ast.get_source_segment(text, node) or ''
    return ''


def _equations(name: str) -> str | None:
    """Return the equations of a model as the symbolic layer shows them, or `None` for forms without equations."""
    if describe(name).formalism not in EQUATION_FORMS:
        return None
    try:
        equations = SymbolicAnalysis().check().prepare(create(name)).run().equations_
    except AnalysisError:
        return None
    return '\n'.join(f'{target}: {equation}' for target, equation in equations.items())


def model_page(name: str, title: str, examples: dict[str, Example]) -> str:
    """Return the page of one model: its screenshot, description, references, factory, types, and equations."""
    entry = describe(name)
    system = create(name)
    lines = [
        f'# {title}',
        '',
        entry.description,
        '',
        f'- Name: `{name}`',
        f'- Field: `{entry.field}`',
        f'- Formalism: {entry.formalism}',
        f'- References: {"; ".join(entry.references)}',
        '',
        *_gallery_lines(name, examples),
        '## Factory',
        '',
        '```python',
        inspect.getsource(entry.factory).rstrip(),
        '```',
    ]
    sources = [source for source in (_class_source(kind) for kind in _component_types(system)) if source]
    if sources:
        lines += ['', '## Components', '', '```python', '\n\n'.join(source.rstrip() for source in sources), '```']
    equations = _equations(name)
    if equations is not None:
        lines += ['', '## Equations', '', '```text', equations, '```']
    return '\n'.join(lines) + '\n'


def field_page(field: str, titles: dict[str, str]) -> str:
    """Return the index page of a field: the link and the description of every registered model, by subfield."""
    entries = sorted(catalog(field=field), key=lambda entry: titles[entry.name])
    lines = [
        f'# {_title(field)}',
        '',
        f'The registered models of `skcomplex.models.{field}`. The page of each shows its factory, its component types',
        'with their states, parameters, and laws as written, and, for the forms that have them, its equations as',
        '`SymbolicAnalysis` returns them, so it serves as the pattern for a model of your own.',
    ]
    for subfield in sorted({entry.field for entry in entries}):
        lines += ['', f'## {_title(subfield.partition("/")[2])}', '']
        lines += [
            f'- [{titles[entry.name]}]({subfield.partition("/")[2]}/{entry.name}.md): {entry.description}'
            for entry in entries
            if entry.field == subfield
        ]
    return '\n'.join(lines) + '\n'


def index_page(inventory: dict[str, Any]) -> str:
    """Return the landing page of the models: the link and the number of registered models of every field."""
    lines = ['# Models', '', 'The registered models of `skcomplex.models`, by field and subfield.', '']
    lines += [
        f'- [{_title(field)}]({field}/index.md): {len(catalog(field=field))} models'
        for field in inventory
        if catalog(field=field)
    ]
    return '\n'.join(lines) + '\n'


def main() -> None:
    """Write the page of every model, the index of every field, and their navigation, after checking the inventory.

    Raises:
        InventoryError:
            The inventory disagrees with the catalog or breaks its own rules.
    """
    problems = inventory_problems()
    if problems:
        error_msg = 'The model inventory has problems. Fix each in docs/overview/inventory.toml:\n- ' + '\n- '.join(
            problems,
        )
        raise InventoryError(error_msg)
    inventory = _inventory()
    titles = _titles(inventory)
    examples = _examples()
    nav = mkdocs_gen_files.nav.Nav()
    nav[('Models',)] = 'index.md'
    with mkdocs_gen_files.open('models/index.md', 'w') as file:
        file.write(index_page(inventory))
    for field in inventory:
        entries = sorted(catalog(field=field), key=lambda entry: (entry.field, titles[entry.name]))
        if not entries:
            continue
        nav[(_title(field),)] = f'{field}/index.md'
        with mkdocs_gen_files.open(f'models/{field}/index.md', 'w') as file:
            file.write(field_page(field, titles))
        for entry in entries:
            page = f'{entry.field}/{entry.name}.md'
            nav[(_title(field), _title(entry.field.partition('/')[2]), titles[entry.name])] = page
            with mkdocs_gen_files.open(f'models/{page}', 'w') as file:
                file.write(model_page(entry.name, titles[entry.name], examples))
    with mkdocs_gen_files.open('models/SUMMARY.md', 'w') as file:
        file.writelines(nav.build_literate_nav())


main()
