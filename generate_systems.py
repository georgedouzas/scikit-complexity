"""Generate one page per system with its pattern, one index per field, the catalog, and the navigation."""

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

from skcomplex.core.analysis import SymbolicAnalysis
from skcomplex.core.components.elements import BaseElement
from skcomplex.core.components.interactions import BaseInteraction
from skcomplex.core.components.system import System
from skcomplex.core.errors import AnalysisError
from skcomplex.systems import SystemListing, catalog, create_system, describe

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
    """Return the title of every system of the inventory, by name."""
    return {
        name: entry['title']
        for subfields in inventory.values()
        for systems in subfields.values()
        for name, entry in systems.items()
    }


def _title(name: str) -> str:
    """Return the title of a field or a subfield."""
    return name.replace('_', ' ').capitalize()


def _examples() -> dict[str, Example]:
    """Return the gallery example of every system the build selects."""
    return {example.system: example for example in GALLERY.selected()}


def _page_of(example: Example) -> str:
    """Return the path of the gallery page of an example, relative to the documentation folder."""
    return (GALLERY.generated_dir(example) / f'{example.path.stem}.md').relative_to(DOCS).as_posix()


def _gallery_lines(name: str, examples: dict[str, Example]) -> list[str]:
    """Return the screenshot of a system and the link to its example, for a page three folders below the docs."""
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


def _component_lines(kinds: list[type]) -> list[str]:
    """Return the element and interaction classes of a system, each with its nearest class of the core."""
    lines = []
    for title, base in (('Elements', BaseElement), ('Interactions', BaseInteraction)):
        found = [kind for kind in kinds if issubclass(kind, base)]
        if not found:
            continue
        lines += ['', f'{title}:', '']
        for kind in found:
            core = next(item for item in kind.__mro__ if item.__module__.startswith('skcomplex.core.components.'))
            lines.append(f'- `{kind.__name__}`' + ('' if core is kind else f', from `{core.__name__}`'))
    return lines


def _selection_lines(name: str) -> list[str]:
    """Return the block that selects a system by name, reads its description, and runs it."""
    return [
        '## Select this system',
        '',
        '```python',
        'from skcomplex.systems import create_simulation, create_system, describe',
        '',
        f"system = create_system('{name}')",
        f"entry = describe('{name}')",
        f"trajectory = create_simulation('{name}').prepare().run().trajectory_",
        '```',
        '',
    ]


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
    """Return the equations of a system as the symbolic layer shows them, or `None` for forms without equations."""
    if describe(name).formalism not in EQUATION_FORMS:
        return None
    try:
        equations = SymbolicAnalysis().prepare(create_system(name)).run().equations_
    except AnalysisError:
        return None
    return '\n'.join(f'{target}: {equation}' for target, equation in equations.items())


def system_page(name: str, title: str, examples: dict[str, Example]) -> str:
    """Return the page of one system: its screenshot, description, references, factory, types, and equations."""
    entry = describe(name)
    system = create_system(name)
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
        *_selection_lines(name),
        '## Factory',
        '',
        '```python',
        inspect.getsource(entry.factory).rstrip(),
        '```',
    ]
    kinds = _component_types(system)
    sources = [source for source in (_class_source(kind) for kind in kinds) if source]
    lines += ['', '## Components', *_component_lines(kinds)]
    if sources:
        lines += ['', '```python', '\n\n'.join(source.rstrip() for source in sources), '```']
    equations = _equations(name)
    if equations is not None:
        lines += ['', '## Equations', '', '```text', equations, '```']
    return '\n'.join(lines) + '\n'


def field_page(field: str, titles: dict[str, str]) -> str:
    """Return the index page of a field: the link and the description of every registered system, by subfield."""
    entries = sorted(catalog(field=field), key=lambda entry: titles[entry.name])
    lines = [
        f'# {_title(field)}',
        '',
        f'The registered systems of `skcomplex.systems.{field}`. The page of each shows its factory, its component',
        'types with their states, parameters, and laws as written, and, for the forms that have them, its equations as',
        '`SymbolicAnalysis` returns them, so it serves as the pattern for a system of your own.',
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
    """Return the landing page of the systems: the link and the number of registered systems of every field."""
    lines = [
        '# Systems',
        '',
        'The registered systems of `skcomplex.systems`, by field and subfield. The [catalog](catalog.md) lists every',
        'system by field and by tag.',
        '',
    ]
    lines += [
        f'- [{_title(field)}]({field}/index.md): {len(catalog(field=field))} systems'
        for field in inventory
        if catalog(field=field)
    ]
    return '\n'.join(lines) + '\n'


def _catalog_entry(entry: SystemListing, titles: dict[str, str]) -> str:
    """Return the line of a system in the catalog: its link, its name, and its description."""
    return f'- [{titles[entry.name]}]({entry.field}/{entry.name}.md) `{entry.name}`: {entry.description}'


def catalog_page(inventory: dict[str, Any], titles: dict[str, str]) -> str:
    """Return the catalog: every registered system by field and subfield, and again by tag.

    Args:
        inventory:
            The inventory by field and subfield.

        titles:
            The title of every system, by name.

    Returns:
        page:
            The Markdown page.
    """
    entries = [entry for field in inventory for entry in catalog(field=field)]
    lines = [
        '# Catalog',
        '',
        f'All {len(entries)} registered systems, by field and by tag. Select one by its name with `create_system` or',
        '`create_simulation` of `skcomplex.systems`, and read what it needs with `describe`. The search box of the',
        'site finds a system by its name, description, or tag.',
        '',
        '```python',
        'from skcomplex.systems import catalog, create_simulation, create_system, describe',
        '',
        "names = [entry.name for entry in catalog(field='biology/ecology')]",
        "entry = describe('lotka_volterra')",
        "system = create_system('lotka_volterra')",
        '```',
        '',
        '## By field',
    ]
    for field in sorted({entry.field for entry in entries}):
        listed = sorted((entry for entry in entries if entry.field == field), key=lambda entry: titles[entry.name])
        lines += ['', f'### {field}', '', *(_catalog_entry(entry, titles) for entry in listed)]
    lines += ['', '## By tag']
    for tag in sorted({tag for entry in entries for tag in entry.tags}):
        tagged = sorted((entry for entry in entries if tag in entry.tags), key=lambda entry: titles[entry.name])
        lines += ['', f'### {tag}', '', *(_catalog_entry(entry, titles) for entry in tagged)]
    return '\n'.join(lines) + '\n'


def main() -> None:
    """Write the page of every system, the index of every field, the catalog, and the navigation, after the inventory.

    Raises:
        InventoryError:
            The inventory disagrees with the catalog or breaks its own rules.
    """
    problems = inventory_problems()
    if problems:
        error_msg = 'The system inventory has problems. Fix each in docs/overview/inventory.toml:\n- ' + '\n- '.join(
            problems,
        )
        raise InventoryError(error_msg)
    inventory = _inventory()
    titles = _titles(inventory)
    examples = _examples()
    nav = mkdocs_gen_files.nav.Nav()
    nav[('Systems',)] = 'index.md'
    nav[('Catalog',)] = 'catalog.md'
    with mkdocs_gen_files.open('systems/index.md', 'w') as file:
        file.write(index_page(inventory))
    with mkdocs_gen_files.open('systems/catalog.md', 'w') as file:
        file.write(catalog_page(inventory, titles))
    for field in inventory:
        entries = sorted(catalog(field=field), key=lambda entry: (entry.field, titles[entry.name]))
        if not entries:
            continue
        nav[(_title(field),)] = f'{field}/index.md'
        with mkdocs_gen_files.open(f'systems/{field}/index.md', 'w') as file:
            file.write(field_page(field, titles))
        for entry in entries:
            page = f'{entry.field}/{entry.name}.md'
            nav[(_title(field), _title(entry.field.partition('/')[2]), titles[entry.name])] = page
            with mkdocs_gen_files.open(f'systems/{page}', 'w') as file:
                file.write(system_page(entry.name, titles[entry.name], examples))
    with mkdocs_gen_files.open('systems/SUMMARY.md', 'w') as file:
        file.writelines(nav.build_literate_nav())


main()
