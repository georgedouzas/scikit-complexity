"""Generate one page per system with its pattern, one index per field, the catalog, and the navigation."""

import ast
import inspect
import sys
import tomllib
from importlib.resources import files
from pathlib import Path
from typing import Any

import mkdocs_gen_files
from gallery_tools import Example, build_from_environment, library_systems
from inventory_checks import InventoryError, inventory_problems

from skcomplex.core.analysis import symbolic
from skcomplex.core.components.elements import BaseElement
from skcomplex.core.components.interactions import BaseInteraction
from skcomplex.core.components.system import BaseSystem, System
from skcomplex.core.errors import AnalysisError

DOCS = Path(str(files('gallery_tools'))).parent
INVENTORY = DOCS / 'overview' / 'inventory.toml'
GALLERY = build_from_environment(DOCS)
EQUATION_LAWS = {'rate', 'noise', 'update'}


def _listing(field: str | None = None) -> list[tuple[str, type[BaseSystem]]]:
    """Return the gallery name and the class of every ready-made system, optionally of a field, sorted by name."""
    return [
        (name, system)
        for name, system in sorted(library_systems().items())
        if field is None or system.field == field or system.field.startswith(f'{field}/')
    ]


def _description(system: type[BaseSystem]) -> str:
    """Return the description of a system class, the first line of its docstring."""
    return (inspect.getdoc(system) or '').split('\n', maxsplit=1)[0]


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


def _selection_lines(system: type[BaseSystem]) -> list[str]:
    """Return the block that imports a system class and simulates it with its tuned run settings."""
    return [
        '## Use this system',
        '',
        '```python',
        'from skcomplex.core.engine.runtime import simulate',
        f'from {system.__module__.rpartition(".")[0]} import {system.__name__}',
        '',
        f'trajectory = simulate({system.__name__}())',
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


def _equations(system: BaseSystem) -> str | None:
    """Return the equations of a system as the symbolic layer shows them, or `None` for forms without equations."""
    tags = system.get_tags()
    if tags.space != {'none'} or not tags.laws or not tags.laws <= EQUATION_LAWS:
        return None
    try:
        equations = symbolic(system.prepare().system_).equations
    except AnalysisError:
        return None
    return '\n'.join(f'{target}: {equation}' for target, equation in equations.items())


def system_page(name: str, title: str, examples: dict[str, Example]) -> str:
    """Return the page of one system: its screenshot, description, references, class, types, and equations."""
    system_class = library_systems()[name]
    built = system_class()
    lines = [
        f'# {title}',
        '',
        _description(system_class),
        '',
        f'- Class: `{system_class.__name__}`',
        f'- Field: `{system_class.field}`',
        f'- References: {"; ".join(system_class.references)}',
        '',
        *_gallery_lines(name, examples),
        *_selection_lines(system_class),
        '## Class',
        '',
        '```python',
        inspect.getsource(system_class).rstrip(),
        '```',
    ]
    kinds = _component_types(built.prepare().system_)
    sources = [source for source in (_class_source(kind) for kind in kinds) if source]
    lines += ['', '## Components', *_component_lines(kinds)]
    if sources:
        lines += ['', '```python', '\n\n'.join(source.rstrip() for source in sources), '```']
    equations = _equations(built)
    if equations is not None:
        lines += ['', '## Equations', '', '```text', equations, '```']
    return '\n'.join(lines) + '\n'


def field_page(field: str, titles: dict[str, str]) -> str:
    """Return the index page of a field: the link and the description of every ready-made system, by subfield."""
    entries = sorted(_listing(field), key=lambda entry: titles[entry[0]])
    lines = [
        f'# {_title(field)}',
        '',
        f'The ready-made systems of `skcomplex.systems.{field}`. The page of each shows its class, its component',
        'types with their states, parameters, and laws as written, and, for the forms that have them, its equations as',
        '`symbolic` returns them, so it serves as the pattern for a system of your own.',
    ]
    for subfield in sorted({system.field for _, system in entries}):
        lines += ['', f'## {_title(subfield.partition("/")[2])}', '']
        lines += [
            f'- [{titles[name]}]({subfield.partition("/")[2]}/{name}.md): {_description(system)}'
            for name, system in entries
            if system.field == subfield
        ]
    return '\n'.join(lines) + '\n'


def index_page(inventory: dict[str, Any]) -> str:
    """Return the landing page of the systems: the link and the number of ready-made systems of every field."""
    lines = [
        '# Systems',
        '',
        'The ready-made systems of `skcomplex.systems`, by field and subfield. The [catalog](catalog.md) lists every',
        'system by field and by topic.',
        '',
    ]
    lines += [
        f'- [{_title(field)}]({field}/index.md): {len(_listing(field))} systems'
        for field in inventory
        if _listing(field)
    ]
    return '\n'.join(lines) + '\n'


def _catalog_entry(name: str, system: type[BaseSystem], titles: dict[str, str]) -> str:
    """Return the line of a system in the catalog: its link, its class, and its description."""
    return f'- [{titles[name]}]({system.field}/{name}.md) `{system.__name__}`: {_description(system)}'


def catalog_page(inventory: dict[str, Any], titles: dict[str, str]) -> str:
    """Return the catalog: every ready-made system by field and subfield, and again by topic.

    Args:
        inventory:
            The inventory by field and subfield.

        titles:
            The title of every system, by name.

    Returns:
        page:
            The Markdown page.
    """
    entries = [entry for field in inventory for entry in _listing(field)]
    lines = [
        '# Catalog',
        '',
        'Every ready-made system, by field and by topic. Import a system class from its subfield, or list the',
        'classes with `all_systems` of `skcomplex.systems`. The search box of the site finds a system by its name,',
        'description, or topic.',
        '',
        '```python',
        'from skcomplex.systems import all_systems',
        '',
        "names = [name for name, _ in all_systems(field='biology/ecology')]",
        '```',
        '',
        '## By field',
    ]
    for field in sorted({system.field for _, system in entries}):
        listed = sorted((entry for entry in entries if entry[1].field == field), key=lambda entry: titles[entry[0]])
        lines += ['', f'### {field}', '', *(_catalog_entry(name, system, titles) for name, system in listed)]
    lines += ['', '## By topic']
    for topic in sorted({topic for _, system in entries for topic in system.topics}):
        tagged = sorted((entry for entry in entries if topic in entry[1].topics), key=lambda entry: titles[entry[0]])
        lines += ['', f'### {topic}', '', *(_catalog_entry(name, system, titles) for name, system in tagged)]
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
        entries = sorted(_listing(field), key=lambda entry: (entry[1].field, titles[entry[0]]))
        if not entries:
            continue
        nav[(_title(field),)] = f'{field}/index.md'
        with mkdocs_gen_files.open(f'systems/{field}/index.md', 'w') as file:
            file.write(field_page(field, titles))
        for name, system in entries:
            page = f'{system.field}/{name}.md'
            nav[(_title(field), _title(system.field.partition('/')[2]), titles[name])] = page
            with mkdocs_gen_files.open(f'systems/{page}', 'w') as file:
                file.write(system_page(name, titles[name], examples))
    with mkdocs_gen_files.open('systems/SUMMARY.md', 'w') as file:
        file.writelines(nav.build_literate_nav())


main()
