"""Generate the coverage page of the model inventory and one page per field with the pattern of every model."""

import ast
import inspect
import sys
import tomllib
from pathlib import Path
from typing import Any

from skcomplex.core.analysis import equations_of
from skcomplex.core.system import System
from skcomplex.models import catalog, create, describe

sys.path.insert(0, str(Path(__file__).parent))

from gallery_tools import Example, build_from_environment, find_examples

DOCS = Path(__file__).parent
INVENTORY = DOCS / 'overview' / 'inventory.toml'
GALLERY = build_from_environment(DOCS)
EQUATION_FORMS = {
    'ordinary differential equations',
    'stochastic differential equations',
    'delay differential equations',
    'discrete map',
}
SHOWN_MODELS = 8
STATUS_LABELS = {'implemented': 'Implemented', 'pending': 'Pending', 'blocked': 'Blocked'}


def _load() -> tuple[dict[str, str], dict[str, Any]]:
    """Return the source keys and the inventory by field and subfield."""
    data = tomllib.loads(INVENTORY.read_text())
    data.pop('primitives')
    data.pop('vocabulary')
    return data.pop('sources'), data


def _vocabulary() -> dict[str, dict[str, str]]:
    """Return the primitive vocabulary of the inventory, by name."""
    return tomllib.loads(INVENTORY.read_text())['primitives']


def _title(name: str) -> str:
    """Return the title of a field or a subfield."""
    return name.replace('_', ' ').capitalize()


def _source(source: str, keys: dict[str, str]) -> str:
    """Return a source with its key expanded to the reference it stands for."""
    key, _, section = source.partition(' ')
    return f'{keys[key]} {section}'.strip() if key in keys else source


def _status(entry: dict[str, Any]) -> str:
    """Return the status of an entry as Markdown, naming the change it waits for."""
    status, _, change = entry['status'].partition(' ')
    if status in {'pending', 'blocked'}:
        detail = f': {entry["feature"]}' if status == 'blocked' else ''
        return f'{STATUS_LABELS[status]} on `{change}`{detail}'
    return STATUS_LABELS[status]


def _examples() -> dict[str, Example]:
    """Return the gallery example of every model that has one."""
    return {example.model: example for example in find_examples(GALLERY.examples)}


def _page_of(example: Example) -> str:
    """Return the path of the gallery page of an example, relative to the documentation folder."""
    return (GALLERY.generated_dir(example) / f'{example.path.stem}.md').relative_to(DOCS).as_posix()


def _gallery_lines(name: str, examples: dict[str, Example]) -> list[str]:
    """Return the screenshot of a model and the link to its example, for a page one folder below the documentation."""
    example = examples.get(name)
    if example is None:
        return []
    page = f'../{_page_of(example)}'
    screenshot = GALLERY.screenshot(example)
    lines = [f'[Example]({page})', '']
    if screenshot.exists():
        image = f'../{screenshot.relative_to(DOCS).as_posix()}'
        lines = [f'[![The {name} example]({image}){{ width="480" }}]({page})', '', *lines]
    return lines


def _primitive_users(inventory: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    """Return the links of the implemented models of every primitive, and the changes its other entries wait for."""
    users: dict[str, list[str]] = {name: [] for name in _vocabulary()}
    waiting: dict[str, set[str]] = {name: set() for name in users}
    entries = [
        (field, name, entry)
        for field, subfields in inventory.items()
        for models in subfields.values()
        for name, entry in models.items()
    ]
    for field, name, entry in entries:
        status, _, change = entry['status'].partition(' ')
        for primitive in entry['primitives']:
            if status == 'implemented':
                users[primitive].append(f'[{name}]({field}.md#{name})')
            else:
                waiting[primitive].add(change)
    return users, waiting


def _coverage(models: list[str], waiting: set[str], change: str | None) -> str:
    """Return the coverage of one primitive from its implemented models and the changes that add or need it."""
    if models:
        return f'Covered by {len(models)}'
    coverage = f'Uncovered, added by `{change}`' if change else 'Uncovered'
    if waiting - {change}:
        coverage += ', waited for by ' + ', '.join(f'`{name}`' for name in sorted(waiting))
    return coverage


def primitive_lines(inventory: dict[str, Any]) -> list[str]:
    """Return the coverage of every primitive: the implemented models that exercise it, or the change that adds it.

    Args:
        inventory:
            The inventory by field and subfield.

    Returns:
        lines:
            The Markdown lines of the section.
    """
    users, waiting = _primitive_users(inventory)
    lines = [
        '',
        '## Coverage by primitive',
        '',
        'Models exist first to exercise the core. Every entry records the core primitives it exercises. Its priority',
        'is `core` when it exercises a primitive or a combination that no implemented model exercises, or needs a',
        'missing one, and `example` when it repeats a covered pattern. A primitive is uncovered when no implemented',
        'model exercises it.',
        '',
        '| Primitive | Description | Coverage | Implemented models |',
        '|---|---|---|---|',
    ]
    for primitive, item in _vocabulary().items():
        models = users[primitive]
        shown = ', '.join(models[:SHOWN_MODELS])
        if len(models) > SHOWN_MODELS:
            shown += f', and {len(models) - SHOWN_MODELS} more'
        coverage = _coverage(models, waiting[primitive], item.get('change'))
        lines.append(f'| `{primitive}` | {item["description"]} | {coverage} | {shown} |')
    return lines


def system_lines(inventory: dict[str, Any]) -> list[str]:
    """Return the number of implemented and pending entries for every term of the system vocabulary.

    Args:
        inventory:
            The inventory by field and subfield.

    Returns:
        lines:
            The Markdown lines of the section.
    """
    vocabulary = tomllib.loads(INVENTORY.read_text())['vocabulary']
    entries = [entry for subfields in inventory.values() for models in subfields.values() for entry in models.values()]
    lines = [
        '',
        '## Coverage by system vocabulary',
        '',
        'A system is its components, interactions, space, and time, and its dynamics are laws. Every entry maps to',
        'this vocabulary. A term no implemented entry uses is a general primitive the core still needs.',
        '',
        '| Category | Term | Description | Implemented | Pending |',
        '|---|---|---|---|---|',
    ]
    for category, terms in vocabulary.items():
        for term, item in terms.items():
            description = item['description'] if isinstance(item, dict) else item
            users = [entry['status'].split(' ')[0] for entry in entries if term in entry['system'][category]]
            implemented = users.count('implemented')
            lines.append(f'| {category} | `{term}` | {description} | {implemented} | {len(users) - implemented} |')
    return lines


def coverage_page() -> str:
    """Return the coverage page: the inventory grouped by field and subfield, with sources and status."""
    keys, inventory = _load()
    examples = _examples()
    lines = [
        '# Model coverage',
        '',
        'Every model surveyed from the reference books and articles, with its sources and status. Implemented models',
        'link to the page of their field, which shows how each is built from the core interface.',
        '',
        '| Field | Implemented | Pending | Blocked |',
        '|---|---|---|---|',
    ]
    for field, subfields in inventory.items():
        statuses = [entry['status'].split(' ')[0] for models in subfields.values() for entry in models.values()]
        counts = [statuses.count(status) for status in STATUS_LABELS]
        lines.append(f'| [{_title(field)}](#{field.replace("_", "-")}) | {" | ".join(map(str, counts))} |')
    lines += system_lines(inventory)
    lines += primitive_lines(inventory)
    for field, subfields in inventory.items():
        lines += ['', f'## {_title(field)} {{#{field.replace("_", "-")}}}']
        for subfield, models in subfields.items():
            header = ['| Model | Sources | Primitives | Priority | Status | Example |', '|---|---|---|---|---|---|']
            lines += ['', f'### {field}/{subfield}', '', *header]
            for name, entry in models.items():
                model, example = entry['title'], ''
                if entry['status'] == 'implemented':
                    model = f'[{model}]({field}.md#{name})'
                if name in examples:
                    example = f'[{examples[name].path.stem}](../{_page_of(examples[name])})'
                sources = '; '.join(_source(source, keys) for source in entry['sources'])
                primitives = ', '.join(f'`{primitive}`' for primitive in entry['primitives'])
                cells = [model, sources, primitives, entry['priority'], _status(entry), example]
                lines.append(f'| {" | ".join(cells)} |')
    return '\n'.join(lines) + '\n'


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
    """Return the source of a component type, a class statement or an assignment of `define`."""
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
    return '\n'.join(f'{target}: {equation}' for target, equation in equations_of(create(name)).items())


def model_section(name: str, examples: dict[str, Example]) -> str:
    """Return the section of one model: its screenshot, description, references, factory, types, and equations."""
    entry = describe(name)
    system = create(name)
    lines = [
        f'## {name}',
        '',
        entry.description,
        '',
        f'- Field: `{entry.field}`',
        f'- Formalism: {entry.formalism}',
        f'- References: {"; ".join(entry.references)}',
        '',
        *_gallery_lines(name, examples),
        '### Factory',
        '',
        '```python',
        inspect.getsource(entry.factory).rstrip(),
        '```',
    ]
    sources = [source for source in (_class_source(kind) for kind in _component_types(system)) if source]
    if sources:
        lines += ['', '### Components', '', '```python', '\n\n'.join(source.rstrip() for source in sources), '```']
    equations = _equations(name)
    if equations is not None:
        lines += ['', '### Equations', '', '```text', equations, '```']
    return '\n'.join(lines)


def field_page(field: str) -> str:
    """Return the page of a field: the pattern of every registered model of the field."""
    entries = catalog(field=field)
    examples = _examples()
    lines = [
        f'# {_title(field)}',
        '',
        f'The registered models of `skcomplex.models.{field}`. Each shows its factory, its component types with their',
        'states, parameters, and laws as written, and, for the forms that have them, its equations as `equations_of`',
        'returns them, so it serves as the pattern for a model of your own.',
    ]
    for subfield in sorted({entry.field for entry in entries}):
        lines += ['', f'# {subfield}', '']
        lines += [model_section(entry.name, examples) + '\n' for entry in entries if entry.field == subfield]
    return '\n'.join(lines)


def fields() -> list[str]:
    """Return the fields of the inventory in order."""
    return list(_load()[1])


def main() -> None:
    """Write the coverage page and the field pages into the documentation."""
    import mkdocs_gen_files  # noqa: PLC0415

    with mkdocs_gen_files.open('models/coverage.md', 'w') as file:
        file.write(coverage_page())
    for field in fields():
        if catalog(field=field):
            with mkdocs_gen_files.open(f'models/{field}.md', 'w') as file:
                file.write(field_page(field))


main()
