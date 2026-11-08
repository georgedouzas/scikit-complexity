"""Generate the system pages, the field indexes, the catalog, and the navigation.

There is one page per system with its pattern, and one index per field.
"""

import ast
import inspect
import sys
import tomllib
from collections.abc import Iterable
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
from skcomplex.core.specs import Tags

DOCS = Path(str(files('gallery_tools'))).parent
INVENTORY = DOCS / 'overview' / 'inventory.toml'
GALLERY = build_from_environment(DOCS)
EQUATION_LAWS = {'rate', 'noise', 'update'}
DEFAULT_WIDTH = 60
TAG_FIELDS = ('time', 'space', 'laws', 'randomness', 'domains', 'structure')


def _listing(field: str | None = None) -> list[tuple[str, type[BaseSystem]]]:
    """Return the gallery name and the class of every ready-made system.

    The systems are optionally of a field, sorted by name.
    """
    return [
        (name, system)
        for name, system in sorted(library_systems().items())
        if field is None or system.field == field or system.field.startswith(f'{field}/')
    ]


_BUILT: dict[str, BaseSystem] = {}


def _built(system: type[BaseSystem]) -> BaseSystem:
    """Return the system a class builds with its default parameters.

    It is built once for its tags and its page.
    """
    key = f'{system.__module__}.{system.__qualname__}'
    if key not in _BUILT:
        _BUILT[key] = system()
    return _BUILT[key]


def _tags(system: type[BaseSystem]) -> Tags:
    """Return the capability tags of a system class.

    They are those of the system it builds with its default parameters.
    """
    return _built(system).get_tags()


def _import_line(system: type[BaseSystem]) -> str:
    """Return the statement that imports a system class from its subfield."""
    return f'from {system.__module__.rpartition(".")[0]} import {system.__name__}'


def _parameters(system: type[BaseSystem]) -> list[tuple[str, str]]:
    """Return the name and the default of every model parameter of a system class.

    The label is left out.
    """
    parameters = list(inspect.signature(system.__init__).parameters.values())[1:]
    variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    return [
        (parameter.name, 'required' if parameter.default is parameter.empty else _default(parameter.default))
        for parameter in parameters
        if parameter.name != 'label' and parameter.kind not in variadic
    ]


def _default(value: object) -> str:
    """Return the default of a parameter as a code span of one line.

    It is shortened when it is long.
    """
    text = ' '.join(repr(value).split())
    if len(text) > DEFAULT_WIDTH:
        text = f'{text[: DEFAULT_WIDTH - 3]}...'
    return f'`{text}`'.replace('|', '\\|')


def _tag_values(values: Iterable[str] | str | None, empty: str = 'any') -> str:
    """Return the values of a tag as sorted code spans.

    They are `empty` for none, and `unknown` when the tag is not declared.
    """
    if values is None:
        return 'unknown'
    found = [values] if isinstance(values, str) else sorted(values)
    return ', '.join(f'`{value}`' for value in found) or empty


def _tag_cell(tags: Tags, name: str) -> str:
    """Return the values of one capability tag, an empty structure read as fixed."""
    return _tag_values(getattr(tags, name), empty='fixed' if name == 'structure' else 'any')


def _table(header: list[str], rows: Iterable[list[str]]) -> list[str]:
    """Return the Markdown lines of a table."""
    lines = [f'| {" | ".join(header)} |', f'|{"---|" * len(header)}']
    return lines + [f'| {" | ".join(row)} |' for row in rows]


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
    """Return the path of the gallery page of an example.

    It is relative to the documentation folder.
    """
    return (GALLERY.generated_dir(example) / f'{example.path.stem}.md').relative_to(DOCS).as_posix()


def _gallery_lines(name: str, examples: dict[str, Example]) -> list[str]:
    """Return the screenshot of a system and the link to its example.

    They are for a page three folders below the docs.
    """
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
    """Return the element and interaction types of a system and its subsystems.

    They are in order of first use.
    """
    types: list[type] = []
    for component in [*system.elements, *system.interactions]:
        kind = getattr(component, 'element', None) or type(component)
        if kind not in types:
            types.append(kind)
    for subsystem in system.systems:
        types += [kind for kind in _component_types(subsystem) if kind not in types]
    return types


def _component_lines(kinds: list[type]) -> list[str]:
    """Return the element and interaction classes of a system.

    Each comes with its nearest class of the core.
    """
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
    """Return the block that imports a system class and simulates it.

    It simulates it with its tuned run settings.
    """
    return [
        '## Use this system',
        '',
        '```python',
        'from skcomplex.core.engine.runtime import simulate',
        _import_line(system),
        '',
        f'trajectory = simulate({system.__name__}())',
        '```',
        '',
    ]


def _class_source(kind: type) -> str:
    """Return the source of an element or interaction type.

    It is its class statement or the assignment that builds it.
    """
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
    """Return the equations of a system as the symbolic layer shows them.

    It returns `None` for forms without equations.
    """
    tags = system.get_tags()
    if tags.space != {'none'} or not tags.laws or not tags.laws <= EQUATION_LAWS:
        return None
    try:
        equations = symbolic(system.prepare().system_).equations
    except AnalysisError:
        return None
    return '\n'.join(f'{target}: {equation}' for target, equation in equations.items())


def _parameter_lines(system: type[BaseSystem]) -> list[str]:
    """Return the model parameters of a system class with their defaults.

    `set_params` changes them.
    """
    parameters = _parameters(system)
    lines = ['## Parameters', '']
    if not parameters:
        return [*lines, 'The system has no model parameter.', '']
    rows = ([f'`{name}`', default] for name, default in parameters)
    return [*lines, *_table(['Parameter', 'Default'], rows), '']


def _tag_lines(system: type[BaseSystem]) -> list[str]:
    """Return the capability tags of a system class.

    `get_tags()` returns them, and solvers and analyses check them.
    """
    tags = _tags(system)
    rows = ([name.capitalize(), _tag_cell(tags, name)] for name in TAG_FIELDS)
    return ['## Tags', '', *_table(['Tag', 'Values'], rows), '']


def system_page(name: str, title: str, examples: dict[str, Example]) -> str:
    """Return the page of one system.

    It holds its screenshot, description, import, parameters, tags, class, and equations.
    """
    system_class = library_systems()[name]
    built = _built(system_class)
    lines = [
        f'# {title}',
        '',
        _description(system_class),
        '',
        f'- Class: `{system_class.__name__}`',
        f'- Import: `{_import_line(system_class)}`',
        f'- Field: `{system_class.field}`',
        f'- Topics: {", ".join(f"`{topic}`" for topic in system_class.topics)}',
        f'- References: {"; ".join(system_class.references)}',
        '',
        *_gallery_lines(name, examples),
        *_selection_lines(system_class),
        *_parameter_lines(system_class),
        *_tag_lines(system_class),
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
    """Return the index page of a field.

    It holds the link and the description of every ready-made system, by subfield.
    """
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
    """Return the landing page of the systems.

    It holds the link and the number of ready-made systems of every field.
    """
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
    """Return the line of a system in the catalog.

    It holds its link, its class, and its description.
    """
    return f'- [{titles[name]}]({system.field}/{name}.md) `{system.__name__}`: {_description(system)}'


def _catalog_row(name: str, system: type[BaseSystem], titles: dict[str, str]) -> list[str]:
    """Return the row of a system in the catalog.

    It holds its link, its class, its parameters, and its capability tags.
    """
    tags = _tags(system)
    parameters = ', '.join(f'`{parameter}`' for parameter, _ in _parameters(system)) or 'none'
    cells = [_tag_cell(tags, tag) for tag in ('time', 'space', 'laws', 'randomness')]
    return [f'[{titles[name]}]({system.field}/{name}.md)', f'`{system.__name__}`', parameters, *cells]


def catalog_page(inventory: dict[str, Any], titles: dict[str, str]) -> str:
    """Return the catalog.

    It lists every ready-made system by field and subfield, and again by topic.

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
        'Every ready-made system, by field and by topic. Each subfield shows the statement that imports its',
        'classes, and each system its class, its model parameters, and its capability tags. The page of a system',
        'gives the defaults of its parameters and all its tags. List the classes of a field, a topic, or a tag with',
        '`all_systems` of `skcomplex.systems`, and find a system by its name, description, or topic with the search',
        'box of the site.',
        '',
        '```python',
        'from skcomplex.systems import all_systems',
        '',
        "names = [name for name, _ in all_systems(field='biology/ecology', tags={'randomness': 'stochastic'})]",
        '```',
        '',
        '## By field',
    ]
    header = ['System', 'Class', 'Parameters', 'Time', 'Space', 'Laws', 'Randomness']
    for field in sorted({system.field for _, system in entries}):
        listed = sorted((entry for entry in entries if entry[1].field == field), key=lambda entry: titles[entry[0]])
        statement = f'from skcomplex.systems.{field.replace("/", ".")} import <Class>'
        rows = (_catalog_row(name, system, titles) for name, system in listed)
        lines += ['', f'### {field}', '', f'`{statement}`', '', *_table(header, rows)]
    lines += ['', '## By topic']
    for topic in sorted({topic for _, system in entries for topic in system.topics}):
        tagged = sorted((entry for entry in entries if topic in entry[1].topics), key=lambda entry: titles[entry[0]])
        lines += ['', f'### {topic}', '', *(_catalog_entry(name, system, titles) for name, system in tagged)]
    return '\n'.join(lines) + '\n'


def main() -> None:
    """Write the system pages, field indexes, catalog, and navigation.

    It writes the page of every system and the index of every field after the inventory.

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
