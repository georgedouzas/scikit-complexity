"""Generate the coverage page of the model inventory and one page per field with the pattern of every model."""

import ast
import inspect
import sys
import tomllib
from pathlib import Path
from typing import Any

from skcomplex.core.analysis import equations_of
from skcomplex.core.model import System
from skcomplex.models import catalog, create, describe

INVENTORY = Path(__file__).parent / 'overview' / 'inventory.toml'
EQUATION_FORMS = {
    'ordinary differential equations',
    'stochastic differential equations',
    'delay differential equations',
    'discrete map',
}
STATUS_LABELS = {'implemented': 'Implemented', 'pending': 'Pending', 'blocked': 'Blocked', 'out': 'Out of scope'}


def _load() -> tuple[dict[str, str], dict[str, Any]]:
    """Return the source keys and the inventory by field and subfield."""
    data = tomllib.loads(INVENTORY.read_text())
    return data.pop('sources'), data


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
    if status == 'out':
        return f'Out of scope: {entry["reason"]}, a future change could wrap {entry["library"]}'
    return STATUS_LABELS[status]


def coverage_page() -> str:
    """Return the coverage page: the inventory grouped by field and subfield, with sources and status."""
    keys, inventory = _load()
    lines = [
        '# Model coverage',
        '',
        'Every model surveyed from the reference books and articles, with its sources and status. Implemented models',
        'link to the page of their field, which shows how each is built from the core interface.',
        '',
        '| Field | Implemented | Pending | Blocked | Out of scope |',
        '|---|---|---|---|---|',
    ]
    for field, subfields in inventory.items():
        statuses = [entry['status'].split(' ')[0] for models in subfields.values() for entry in models.values()]
        counts = [statuses.count(status) for status in STATUS_LABELS]
        lines.append(f'| [{_title(field)}](#{field.replace("_", "-")}) | {" | ".join(map(str, counts))} |')
    for field, subfields in inventory.items():
        lines += ['', f'## {_title(field)} {{#{field.replace("_", "-")}}}']
        for subfield, models in subfields.items():
            lines += ['', f'### {field}/{subfield}', '', '| Model | Sources | Status |', '|---|---|---|']
            for name, entry in models.items():
                model = entry['title']
                if entry['status'] == 'implemented':
                    model = f'[{model}]({field}.md#{name})'
                sources = '; '.join(_source(source, keys) for source in entry['sources'])
                lines.append(f'| {model} | {sources} | {_status(entry)} |')
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


def model_section(name: str) -> str:
    """Return the section of one model: its description, references, factory, component types, and equations."""
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
    lines = [
        f'# {_title(field)}',
        '',
        f'The registered models of `skcomplex.models.{field}`. Each shows its factory, its component types with their',
        'states, parameters, and laws as written, and, for the forms that have them, its equations as `equations_of`',
        'returns them, so it serves as the pattern for a model of your own.',
    ]
    for subfield in sorted({entry.field for entry in entries}):
        lines += ['', f'# {subfield}', '']
        lines += [model_section(entry.name) + '\n' for entry in entries if entry.field == subfield]
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
