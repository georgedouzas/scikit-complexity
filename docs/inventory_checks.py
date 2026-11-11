"""Check the inventory of surveyed systems.

It is checked against the system classes, their tags, its vocabularies, and the changes.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
import tomllib
from importlib.resources import files
from pathlib import Path
from typing import Any

from skcomplex.core.components.systems import BaseSystem
from skcomplex.core.enums import DomainTag, LawTag, SpaceTag, TimeTag
from skcomplex.core.errors import SkcomplexError
from skcomplex.systems import all_systems

ROOT = Path(str(files('gallery_tools'))).parent.parent
INVENTORY = ROOT / 'docs' / 'overview' / 'inventory.toml'
CHANGES = ROOT / 'openspec' / 'changes'
ARXIV = re.compile(r'(\d{4}\.\d{4,5}|[a-z-]+/\d{7})( |$)')
CITATION = re.compile(r'.+ \d{4}( |$)')
INVENTORY_CHANGE = 'organize-system-library'
STATUSES = re.compile(r'implemented|(pending|blocked) [a-z0-9-]+')
PRIORITIES = ('core', 'example')
TAGGED = {'time': TimeTag, 'space': SpaceTag, 'laws': LawTag, 'domains': DomainTag}
CLASS_WORDS = re.compile(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')

type Entry = tuple[str, str, dict[str, Any]]


def _load() -> tuple[dict[str, str], list[Entry], dict[str, Any], dict[str, Any]]:
    """Return the source keys, the entries, and the two vocabularies.

    The entries come with their names and field paths.
    """
    data = tomllib.loads(INVENTORY.read_text())
    keys = data.pop('sources')
    primitives = data.pop('primitives')
    vocabulary = data.pop('vocabulary')
    entries = [
        (name, f'{field}/{subfield}', entry)
        for field, subfields in data.items()
        for subfield, systems in subfields.items()
        for name, entry in systems.items()
    ]
    return keys, entries, primitives, vocabulary


def _systems() -> dict[str, type[BaseSystem]]:
    """Return the class of every ready-made system by its inventory name.

    The inventory name is its class name without `System` in snake case. An acronym stays one word, so `GrayScottSystem`
    is `gray_scott` and `BrusselatorODESystem` is `brusselator_ode`.
    """
    return {CLASS_WORDS.sub('_', name.removesuffix('System')).lower(): cls for name, cls in all_systems()}


def _changes() -> tuple[dict[str, str], set[str]]:
    """Return the specs of every active OpenSpec change, and the archived names.

    The specs are given by name, with the names of the archived changes.
    """
    active = {
        path.name: ' '.join(spec.read_text() for spec in sorted(path.glob('specs/**/*.md')))
        for path in CHANGES.iterdir()
        if path.is_dir() and path.name != 'archive'
    }
    archived = {re.sub(r'^\d{4}-\d{2}-\d{2}-', '', path.name) for path in (CHANGES / 'archive').iterdir()}
    return active, archived


def _shape_problems(entries: list[Entry], keys: dict[str, str]) -> list[str]:
    """Return the entries of a bad shape.

    They have a repeated name, no title, an unknown status, or sources that do not resolve.
    """
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


def _catalog_problems(entries: list[Entry], systems: dict[str, type[BaseSystem]]) -> list[str]:
    """Return the disagreements between the inventory entries and the system classes."""
    registered = {name: cls.field for name, cls in systems.items()}
    problems = []
    recorded = {name for name, _, _ in entries}
    for name, path, entry in entries:
        if entry['status'] != 'implemented':
            continue
        if name not in registered:
            problems.append(f'{name} is marked implemented and no system class has that name')
        elif registered[name] != path:
            problems.append(f'{name} is recorded under {path} and its class has the field {registered[name]}')
    problems += [f'{name} is a system class without an inventory entry' for name in sorted(set(registered) - recorded)]
    return problems


def _change_problems(entries: list[Entry], active: dict[str, str], archived: set[str]) -> list[str]:
    """Return the entries whose status names no existing change or lacks details."""
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
    """Return the entries of unsupported primitives or priority, and unknown changes.

    The vocabulary does not support their primitives or priority.
    """
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
    """Return the blocked and the implemented entries that break the priority rules.

    Blocked entries lack a core primitive of their change, and implemented entries have a wrong priority. The first
    implemented entry with a set of primitives is core, and later entries with the same set are examples.
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


def _tag_problems(name: str, entry: dict[str, Any], system_class: type[BaseSystem] | None) -> list[str]:
    """Return the problems of the categories an entry writes.

    They are the categories that the tags of its system class give. An implemented entry writes none of them, and its
    class must derive its tags with known law kinds. An entry that is pending or blocked writes every one of them, with
    the values of the capability tags.
    """
    system = entry['system']
    if entry['status'] != 'implemented':
        problems = [f'{name} records no {category}' for category in TAGGED if not system.get(category)]
        problems.extend(
            f'{name} names the {category} term {term}, which is no value of the capability tag {category}'
            for category, values in TAGGED.items()
            for term in system.get(category, [])
            if term not in values
        )
        return problems
    problems = [
        f'{name} writes its {category}, which get_tags() of its system class gives, so remove it'
        for category in TAGGED
        if category in system
    ]
    if system_class is None:
        return problems
    try:
        tags = system_class().get_tags()
    except SkcomplexError as error:
        return [*problems, f'{name} has a system class whose tags cannot be derived: {error}']
    if tags.laws is None:
        error_msg = f'{name} has a system class whose components do not show their law kinds, so list their law_classes'
        problems.append(error_msg)
    return problems


def _system_problems(
    entries: list[Entry],
    vocabulary: dict[str, Any],
    systems: dict[str, type[BaseSystem]],
) -> list[str]:
    """Return the entries that do not fit the system vocabulary or the tags.

    They do not map to the system vocabulary, or write what the tags of their class give.
    """
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
        problems.extend(
            f'{name} records the unknown category {category}'
            for category in set(system) - set(vocabulary) - set(TAGGED)
        )
        problems.extend(_tag_problems(name, entry, systems.get(name)))
    return problems


def inventory_problems() -> list[str]:
    """Return every problem of the inventory, one sentence each.

    A consistent inventory gives nothing.

    Returns:
        problems:
            The problems of the entry shapes, the agreement with the system classes, the changes, the primitives, the
            priorities, and the system vocabulary with the tags of the system classes.
    """
    keys, entries, primitives, vocabulary = _load()
    active, archived = _changes()
    systems = _systems()
    return [
        *_shape_problems(entries, keys),
        *_catalog_problems(entries, systems),
        *_change_problems(entries, active, archived),
        *_primitive_problems(entries, primitives, set(active) | archived),
        *_priority_problems(entries, primitives),
        *_system_problems(entries, vocabulary, systems),
    ]


class InventoryError(Exception):
    """The inventory of surveyed systems is not consistent.

    It disagrees with the system classes or breaks its own rules.
    """
