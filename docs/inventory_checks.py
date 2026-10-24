"""Check the inventory of surveyed models against the catalog, its own vocabularies, and the OpenSpec changes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
import tomllib
from pathlib import Path
from typing import Any

from skcomplex.models import catalog

ROOT = Path(__file__).parent.parent
INVENTORY = ROOT / 'docs' / 'overview' / 'inventory.toml'
CHANGES = ROOT / 'openspec' / 'changes'
ARXIV = re.compile(r'(\d{4}\.\d{4,5}|[a-z-]+/\d{7})( |$)')
CITATION = re.compile(r'.+ \d{4}( |$)')
INVENTORY_CHANGE = 'organize-model-library'
STATUSES = re.compile(r'implemented|(pending|blocked) [a-z0-9-]+')
PRIORITIES = ('core', 'example')

type Entry = tuple[str, str, dict[str, Any]]


def _load() -> tuple[dict[str, str], list[Entry], dict[str, Any], dict[str, Any]]:
    """Return the source keys, the entries with their names and field paths, and the two vocabularies."""
    data = tomllib.loads(INVENTORY.read_text())
    keys = data.pop('sources')
    primitives = data.pop('primitives')
    vocabulary = data.pop('vocabulary')
    entries = [
        (name, f'{field}/{subfield}', entry)
        for field, subfields in data.items()
        for subfield, models in subfields.items()
        for name, entry in models.items()
    ]
    return keys, entries, primitives, vocabulary


def _registered() -> dict[str, str]:
    """Return the field path of every registered model of the library, by name."""
    return {entry.name: entry.field for entry in catalog() if not entry.field.startswith(('examples', 'test'))}


def _changes() -> tuple[dict[str, str], set[str]]:
    """Return the specs of every active OpenSpec change by name, and the names of the archived changes."""
    active = {
        path.name: ' '.join(spec.read_text() for spec in sorted(path.glob('specs/**/*.md')))
        for path in CHANGES.iterdir()
        if path.is_dir() and path.name != 'archive'
    }
    archived = {re.sub(r'^\d{4}-\d{2}-\d{2}-', '', path.name) for path in (CHANGES / 'archive').iterdir()}
    return active, archived


def _shape_problems(entries: list[Entry], keys: dict[str, str]) -> list[str]:
    """Return the entries with a repeated name, no title, an unknown status, or sources that do not resolve."""
    names = [name for name, _, _ in entries]
    problems = [f'{name} has more than one entry' for name in sorted({name for name in names if names.count(name) > 1})]
    for name, _, entry in entries:
        if not entry.get('title'):
            problems.append(f'{name} has no title')
        if not STATUSES.fullmatch(entry.get('status', '')):
            problems.append(f'{name} has the status {entry.get("status")!r}, which is not a known status')
        if not entry.get('sources'):
            problems.append(f'{name} names no source')
        problems.extend(
            f'{name} names the source {source!r}, which is no source key, arXiv identifier, or citation'
            for source in entry.get('sources', [])
            if source.split(' ')[0] not in keys and not ARXIV.match(source) and not CITATION.match(source)
        )
    return problems


def _catalog_problems(entries: list[Entry], registered: dict[str, str]) -> list[str]:
    """Return the disagreements between the inventory entries and the registered models."""
    problems = []
    recorded = {name for name, _, _ in entries}
    for name, path, entry in entries:
        if entry['status'] != 'implemented':
            continue
        if name not in registered:
            problems.append(f'{name} is marked implemented and no model of that name is registered')
        elif registered[name] != path:
            problems.append(f'{name} is recorded under {path} and registered under {registered[name]}')
    problems += [f'{name} is registered without an inventory entry' for name in sorted(set(registered) - recorded)]
    return problems


def _change_problems(entries: list[Entry], active: dict[str, str], archived: set[str]) -> list[str]:
    """Return the entries whose status does not name an existing change or lacks its details."""
    problems = []
    for name, _, entry in entries:
        status, _, change = entry['status'].partition(' ')
        if status == 'blocked' and change not in set(active) | archived:
            problems.append(f'{name} is blocked on {change}, which is not an OpenSpec change')
        if status == 'blocked' and not entry.get('feature'):
            problems.append(f'{name} is blocked without naming the missing feature')
        if status == 'pending' and change not in active:
            problems.append(f'{name} is pending {change}, which is not an active OpenSpec change')
        elif status == 'pending' and change != INVENTORY_CHANGE and f'`{name}`' not in active[change]:
            problems.append(f'{name} is pending {change}, whose specs do not list it')
    return problems


def _primitive_problems(entries: list[Entry], primitives: dict[str, Any], changes: set[str]) -> list[str]:
    """Return the entries whose primitives or priority the vocabulary does not support, and the unknown changes."""
    problems = [
        f'The primitive {primitive} is added by {item["change"]}, which is not an OpenSpec change'
        for primitive, item in primitives.items()
        if 'change' in item and item['change'] not in changes
    ]
    for name, _, entry in entries:
        if not entry.get('primitives'):
            problems.append(f'{name} records no primitive')
        problems.extend(
            f'{name} names the primitive {primitive}, which the vocabulary does not define'
            for primitive in entry.get('primitives', [])
            if primitive not in primitives
        )
        if entry.get('priority') not in PRIORITIES:
            problems.append(f'{name} has the priority {entry.get("priority")}, which is not core or example')
    return problems


def _priority_problems(entries: list[Entry], primitives: dict[str, Any]) -> list[str]:
    """Return blocked entries without a core primitive of their change, and implemented entries with a wrong priority.

    The first implemented entry with a set of primitives is core, and later entries with the same set are examples.
    """
    problems, seen = [], set()
    for name, _, entry in entries:
        status, _, change = entry['status'].partition(' ')
        recorded = entry.get('primitives', [])
        if status == 'blocked':
            if entry.get('priority') != 'core':
                problems.append(f'{name} is blocked and has the priority {entry.get("priority")}, not core')
            if not any(primitives.get(primitive, {}).get('change') == change for primitive in recorded):
                problems.append(f'{name} is blocked on {change} and records no primitive that {change} adds')
        if status == 'implemented':
            expected = 'example' if frozenset(recorded) in seen else 'core'
            if entry.get('priority') != expected:
                problems.append(
                    f'{name} has the priority {entry.get("priority")}, and its primitives make it {expected}',
                )
            seen.add(frozenset(recorded))
    return problems


def _system_problems(entries: list[Entry], vocabulary: dict[str, Any]) -> list[str]:
    """Return the entries that do not map to the system vocabulary."""
    problems = []
    for name, _, entry in entries:
        system = entry.get('system')
        if system is None:
            problems.append(f'{name} does not map to the system vocabulary')
            continue
        for category, terms in vocabulary.items():
            if category not in system:
                problems.append(f'{name} records no {category}')
                continue
            problems.extend(
                f'{name} names the {category} term {term}, which the vocabulary does not define, so specify the '
                'general primitive it needs'
                for term in system[category]
                if term not in terms
            )
            if category != 'bindings' and not system[category]:
                problems.append(f'{name} records no {category}')
        problems.extend(f'{name} records the unknown category {category}' for category in set(system) - set(vocabulary))
    return problems


def inventory_problems() -> list[str]:
    """Return every problem of the inventory, one sentence each, or nothing for a consistent inventory.

    Returns:
        problems:
            The problems of the entry shapes, the catalog agreement, the changes, the primitives, the priorities, and
            the system vocabulary.
    """
    keys, entries, primitives, vocabulary = _load()
    active, archived = _changes()
    return [
        *_shape_problems(entries, keys),
        *_catalog_problems(entries, _registered()),
        *_change_problems(entries, active, archived),
        *_primitive_problems(entries, primitives, set(active) | archived),
        *_priority_problems(entries, primitives),
        *_system_problems(entries, vocabulary),
    ]


class InventoryError(Exception):
    """The inventory of surveyed models disagrees with the catalog or breaks its own rules."""
