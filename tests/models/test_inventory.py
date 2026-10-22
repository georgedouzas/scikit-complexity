"""Test the inventory of surveyed models against the catalog and the OpenSpec changes."""

import re
import tomllib
from pathlib import Path

import pytest

from skcomplex.models import catalog

ROOT = Path(__file__).parents[2]
INVENTORY = ROOT / 'docs' / 'overview' / 'inventory.toml'
CHANGES = ROOT / 'openspec' / 'changes'
ARXIV = re.compile(r'(\d{4}\.\d{4,5}|[a-z-]+/\d{7})( |$)')
CITATION = re.compile(r'.+ \d{4}( |$)')
INVENTORY_CHANGE = 'organize-model-library'
STATUSES = re.compile(r'implemented|(pending|blocked) [a-z0-9-]+')
PRIORITIES = ('core', 'example')


def _load():
    """Return the source keys and the entries of the inventory, each with its name and field path."""
    data = tomllib.loads(INVENTORY.read_text())
    keys = data.pop('sources')
    data.pop('primitives')
    data.pop('vocabulary')
    entries = [
        (name, f'{field}/{subfield}', entry)
        for field, subfields in data.items()
        for subfield, models in subfields.items()
        for name, entry in models.items()
    ]
    return keys, entries


def _vocabulary():
    """Return the primitive vocabulary of the inventory, by name."""
    return tomllib.loads(INVENTORY.read_text())['primitives']


def _system_vocabulary():
    """Return the system vocabulary of the inventory, by category and term."""
    return tomllib.loads(INVENTORY.read_text())['vocabulary']


def _registered():
    """Return the field path of every registered model of the library, by name."""
    return {entry.name: entry.field for entry in catalog() if not entry.field.startswith(('examples', 'test'))}


def _changes():
    """Return the specs of every active OpenSpec change by name, and the names of the archived changes."""
    active = {
        path.name: ' '.join(spec.read_text() for spec in sorted(path.glob('specs/**/*.md')))
        for path in CHANGES.iterdir()
        if path.is_dir() and path.name != 'archive'
    }
    archived = {re.sub(r'^\d{4}-\d{2}-\d{2}-', '', path.name) for path in (CHANGES / 'archive').iterdir()}
    return active, archived


def _catalog_problems(entries, registered):
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


def _change_problems(entries, active, archived):
    """Return the entries whose status does not name an existing change or lacks its details.

    The change that introduces the inventory lists its own pending models by the inventory itself, so only the specs of
    other pending changes must list a model by its name.
    """
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


def _primitive_problems(entries, vocabulary, changes):
    """Return the entries whose primitives or priority the vocabulary does not support, and the unknown changes.

    Args:
        entries:
            The name, field path, and table of every entry.

        vocabulary:
            The primitives the inventory defines, by name.

        changes:
            The names of the active and archived OpenSpec changes.

    Returns:
        problems:
            One sentence per problem.
    """
    problems = [
        f'The primitive {primitive} is added by {item["change"]}, which is not an OpenSpec change'
        for primitive, item in vocabulary.items()
        if 'change' in item and item['change'] not in changes
    ]
    for name, _, entry in entries:
        if not entry.get('primitives'):
            problems.append(f'{name} records no primitive')
        problems.extend(
            f'{name} names the primitive {primitive}, which the vocabulary does not define'
            for primitive in entry.get('primitives', [])
            if primitive not in vocabulary
        )
        if entry.get('priority') not in PRIORITIES:
            problems.append(f'{name} has the priority {entry.get("priority")}, which is not core or example')
    return problems


def _system_problems(entries, vocabulary):
    """Return the entries that do not map to the system vocabulary.

    Args:
        entries:
            The name, field path, and table of every entry.

        vocabulary:
            The terms of every category of the system vocabulary.

    Returns:
        problems:
            One sentence per problem.
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
                f'{name} names the {category} term {term}, which the vocabulary does not define; '
                'specify the general primitive it needs'
                for term in system[category]
                if term not in terms
            )
            if category != 'bindings' and not system[category]:
                problems.append(f'{name} records no {category}')
        problems.extend(f'{name} records the unknown category {category}' for category in set(system) - set(vocabulary))
    return problems


def test_inventory_entries_are_well_formed():
    """Test that every entry has a unique name, a title, a known status, and sources that resolve."""
    keys, entries = _load()
    names = [name for name, _, _ in entries]
    assert len(names) == len(set(names))
    for name, _, entry in entries:
        assert entry['title'], name
        assert STATUSES.fullmatch(entry['status']), name
        assert entry['sources'], name
        for source in entry['sources']:
            assert source.split(' ')[0] in keys or ARXIV.match(source) or CITATION.match(source), (name, source)


def test_inventory_agrees_with_catalog():
    """Test that every implemented entry is registered under its field path, and every model has an entry."""
    _, entries = _load()
    assert _catalog_problems(entries, _registered()) == []


def test_inventory_entries_record_known_primitives():
    """Test that every entry records primitives of the vocabulary and a priority, and the vocabulary names changes."""
    _, entries = _load()
    active, archived = _changes()
    assert _primitive_problems(entries, _vocabulary(), set(active) | archived) == []


@pytest.mark.parametrize(
    ('entry', 'message'),
    [
        (
            {'primitives': ['rates', 'telepathy'], 'priority': 'core'},
            'model names the primitive telepathy, which the vocabulary does not define',
        ),
        ({'primitives': [], 'priority': 'core'}, 'model records no primitive'),
        (
            {'primitives': ['rates'], 'priority': 'urgent'},
            'model has the priority urgent, which is not core or example',
        ),
    ],
)
def test_inventory_check_names_primitive_problems(entry, message):
    """Test that the inventory check names an unknown primitive, a missing primitive list, and an unknown priority."""
    vocabulary = {'rates': {'description': 'Rate laws'}}
    assert _primitive_problems([('model', 'physics/mechanics', entry)], vocabulary, set()) == [message]


def test_inventory_entries_map_to_the_system_vocabulary():
    """Test that every implemented and pending entry maps to the time, space, components, laws, and the rest."""
    _, entries = _load()
    assert _system_problems(entries, _system_vocabulary()) == []


@pytest.mark.parametrize(
    ('system', 'message'),
    [
        (
            {'time': ['continuous'], 'laws': ['telepathy']},
            (
                'model names the laws term telepathy, which the vocabulary does not define; '
                'specify the general primitive it needs'
            ),
        ),
        ({'time': [], 'laws': ['rate']}, 'model records no time'),
        ({'time': ['continuous']}, 'model records no laws'),
    ],
)
def test_inventory_check_names_unmapped_entries(system, message):
    """Test that the inventory check names an unknown term, an empty category, and a missing category."""
    vocabulary = {'time': {'continuous': 'Continuous time'}, 'laws': {'rate': {'class': 'continuous'}}}
    assert _system_problems([('model', 'physics/mechanics', {'system': system})], vocabulary) == [message]


def test_inventory_system_of_a_book_model():
    """Test that the Bak-Sneppen entry maps to discrete time on a lattice with one member per site."""
    _, entries = _load()
    system = next(entry for name, _, entry in entries if name == 'bak_sneppen')['system']
    assert system['time'] == ['discrete']
    assert system['space'] == ['lattice']
    assert 'site-group' in system['components']


def test_inventory_primitives_of_a_book_model():
    """Test that the Bak-Sneppen entry lists the lattice and extremal-update primitives it exercises."""
    _, entries = _load()
    entry = next(entry for name, _, entry in entries if name == 'bak_sneppen')
    assert {'lattices', 'extremal-updates'} <= set(entry['primitives'])


def test_inventory_priorities_follow_new_primitives():
    """Test that blocked entries need a primitive of their change with priority core, and repeated sets are examples."""
    _, entries = _load()
    vocabulary = _vocabulary()
    seen = set()
    for name, _, entry in entries:
        status, _, change = entry['status'].partition(' ')
        if status == 'blocked':
            assert entry['priority'] == 'core', name
            assert any(vocabulary[primitive].get('change') == change for primitive in entry['primitives']), name
        if status == 'implemented':
            primitives = frozenset(entry['primitives'])
            assert entry['priority'] == ('example' if primitives in seen else 'core'), name
            seen.add(primitives)


def test_inventory_names_existing_changes():
    """Test that blocked and pending entries name OpenSpec changes and out-of-scope entries name a library."""
    _, entries = _load()
    assert _change_problems(entries, *_changes()) == []


@pytest.mark.parametrize(
    ('entries', 'registered', 'message'),
    [
        (
            [('ghost', 'physics/mechanics', {'status': 'implemented'})],
            {},
            'ghost is marked implemented and no model of that name is registered',
        ),
        ([], {'stray': 'physics/mechanics'}, 'stray is registered without an inventory entry'),
        (
            [('moved', 'physics/mechanics', {'status': 'implemented'})],
            {'moved': 'physics/fluids'},
            'moved is recorded under physics/mechanics and registered under physics/fluids',
        ),
    ],
)
def test_inventory_check_names_catalog_disagreements(entries, registered, message):
    """Test that the inventory check names a missing registration, an unrecorded model, and a moved model."""
    assert _catalog_problems(entries, registered) == [message]


@pytest.mark.parametrize(
    ('entry', 'message'),
    [
        ({'status': 'blocked add-nothing', 'feature': 'Nothing'}, 'is blocked on add-nothing, which is not'),
        ({'status': 'blocked add-something'}, 'is blocked without naming the missing feature'),
        ({'status': 'pending add-archived'}, 'is pending add-archived, which is not an active'),
        ({'status': 'pending add-something'}, 'is pending add-something, whose specs do not list it'),
    ],
)
def test_inventory_check_names_change_problems(entry, message):
    """Test that the inventory check names a blocked or pending entry without an existing change or its details."""
    problems = _change_problems([('model', 'physics/mechanics', entry)], {'add-something': '`other`'}, {'add-archived'})
    assert [problem for problem in problems if message in problem]


def test_inventory_entry_of_a_book_model():
    """Test that the Bak-Sneppen entry names its book sections, its field, and its implemented status."""
    keys, entries = _load()
    path, entry = next((path, entry) for name, path, entry in entries if name == 'bak_sneppen')
    assert path == 'biology/evolution'
    assert entry['sources'] == ['Gr 6.6', 'Th 5.4.4', 'Je 4.2']
    assert [keys[source.split(' ')[0]] for source in entry['sources']] == [
        'Gros 2015',
        'Thurner et al. 2018',
        'Jensen 2023',
    ]
    assert entry['status'] == 'implemented'


def _generator():
    """Return the namespace of the documentation generator of the model pages, without writing the pages."""
    path = ROOT / 'docs' / 'generate_models.py'
    namespace = {'__name__': 'generate_models', '__file__': str(path)}
    source = path.read_text().removesuffix('\nmain()\n')
    exec(compile(source, str(path), 'exec'), namespace)  # noqa: S102  # nosec B102: runs the repository's own script
    return namespace


def test_coverage_page_lists_every_entry():
    """Test that the coverage page lists every surveyed model of a subfield and links the implemented ones."""
    page = _generator()['coverage_page']()
    _, entries = _load()
    assert '### earth/climate' in page
    for name, path, entry in entries:
        assert entry['title'] in page, name
        if entry['status'] == 'implemented':
            assert f'({path.split("/")[0]}.md#{name})' in page, name


def test_coverage_page_flags_uncovered_primitives():
    """Test that a primitive exercised only by blocked models is uncovered and names the change that adds it."""
    page = _generator()['coverage_page']()
    row = next(line for line in page.splitlines() if line.startswith('| `link-fields` |'))
    assert 'Uncovered, added by `add-link-fields`' in row
    covered = next(line for line in page.splitlines() if line.startswith('| `extremal-updates` |'))
    assert 'Covered by 1' in covered
    assert '[bak_sneppen](biology.md#bak_sneppen)' in covered


def test_coverage_page_counts_the_system_vocabulary():
    """Test that the coverage page counts implemented and pending entries for every term of every category."""
    page = _generator()['coverage_page']()
    assert '## Coverage by system vocabulary' in page
    for category, terms in _system_vocabulary().items():
        for term in terms:
            assert f'| {category} | `{term}` |' in page, (category, term)


def test_field_page_shows_the_pattern_of_every_model():
    """Test that the biology page shows the equations of the equation forms and the laws of the others."""
    generator = _generator()
    page = generator['field_page']('biology')
    for entry in catalog(field='biology'):
        section = page[page.index(f'## {entry.name}\n') :].split('\n## ')[0]
        assert '### Factory' in section, entry.name
        if entry.formalism in generator['EQUATION_FORMS']:
            assert '### Equations' in section, entry.name
            assert ' == ' in section, entry.name
        else:
            assert 'def laws' in section or 'laws=' in section, entry.name
