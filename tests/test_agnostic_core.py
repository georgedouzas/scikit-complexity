"""Test that the core is application agnostic: it names no field and imports nothing of the models.

Every public name and module of `skcomplex.core` is a mathematical, modelling, or engineering term. Field terms belong
to `skcomplex.models`, whose realizations the core never imports.
"""

import ast
import re
from pathlib import Path

import pytest

CORE = Path(__file__).parent.parent / 'src' / 'skcomplex' / 'core'
MODELS = 'skcomplex.models'
DYNAMIC_IMPORTS = frozenset({'import_module', '__import__'})

# The reviewed list of field terms that no public name or module of the core may contain. A term matches whole words of
# a name, split at underscores, dots, and CamelCase boundaries, case-insensitively and with plural endings, so `spin`
# matches `SpinChain` but not `spinodal`. A term of several words matches consecutive words, such as `OrderBook`. Add a
# term here whenever a review of the core finds a new one.
#
# Physics terms that are the mathematics itself stay allowed: a generator, a Hamiltonian, a dissipator, a potential, a
# particle, quantum states, entropy production, Langevin and Kramers-Moyal processes, and named integrators such as
# velocity Verlet. Modelling terms such as agent, group, interaction, population, stock and flow, and avalanche stay
# allowed too.
#
# `turtle` is forbidden: turtle graphics is a metaphor for the geometric interpretation of a word, which the core calls
# `WordGeometry` in `core.viz`, and the turtle as a named interpretation of L-systems belongs to the models.
FORBIDDEN_TERMS = (
    # Economics and finance.
    'market',
    'bank',
    'firm',
    'trade',
    'trader',
    'order book',
    'auction',
    'institution',
    'price',
    'wealth',
    'money',
    'income',
    'tax',
    'investor',
    'buyer',
    'seller',
    'consumer',
    'household',
    'portfolio',
    # Social science.
    'segregation',
    'schelling',
    'voter',
    'election',
    'opinion',
    'crowd',
    'pedestrian',
    'traffic',
    'vehicle',
    'city',
    # Ecology and biology.
    'predator',
    'prey',
    'predation',
    'herbivore',
    'species',
    'ecosystem',
    'lotka',
    'volterra',
    'gene',
    'genome',
    'protein',
    'neuron',
    'synapse',
    'brain',
    'boids',
    'flock',
    'herd',
    # Epidemiology.
    'epidemic',
    'pandemic',
    'infection',
    'infected',
    'disease',
    'virus',
    'vaccine',
    # Physics realizations: named forces, potentials, species, and models.
    'gravity',
    'spring',
    'lennard',
    'soft sphere',
    'coulomb',
    'thermostat',
    'qubit',
    'spin',
    'ising',
    'potts',
    'kuramoto',
    'vicsek',
    'sandpile',
    'forest fire',
    'earthquake',
    'planet',
    'galaxy',
    'molecule',
    # Mathematics realizations: rule codes, named rules, drawings, and named groups.
    'wolfram',
    'turtle',
    'free group',
    'heisenberg group',
    'lamplighter',
    'baumslag',
    'solitar',
    'grigorchuk',
)


def _words(name):
    """Return the lowercase words of a name, split at underscores, dots, and CamelCase boundaries."""
    return [
        word.lower()
        for part in re.split(r'[_.\W]+', name)
        for word in re.findall(r'[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|[A-Z]+|\d+', part)
    ]


def _matches(word, term):
    """Check whether a word is a term or one of its plurals."""
    return word in {term, f'{term}s', f'{term}es'}


def _field_terms(name):
    """Return the forbidden field terms a name contains."""
    words = _words(name)
    found = []
    for term in FORBIDDEN_TERMS:
        parts = term.split()
        windows = (words[start : start + len(parts)] for start in range(len(words) - len(parts) + 1))
        if any(all(_matches(word, part) for word, part in zip(window, parts, strict=True)) for window in windows):
            found.append(term)
    return found


def _module_of(path):
    """Return the dotted module of a core source file and whether it is a package's `__init__`."""
    parts = list(path.relative_to(CORE.parent.parent).with_suffix('').parts)
    if parts[-1] == '__init__':
        return '.'.join(parts[:-1]), True
    return '.'.join(parts), False


def _public_names(path):
    """Return the names a package's `__init__` lists in its `__all__`."""
    for node in ast.parse(path.read_text()).body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, 'target', None)]
        if any(isinstance(target, ast.Name) and target.id == '__all__' for target in targets):
            return list(ast.literal_eval(node.value))
    return []


def _imported_modules(source, module, *, is_package):
    """Return every module a source imports, with relative imports resolved and `from a import b` read as `a.b`.

    Statements are read from the syntax tree, so doctest lines inside docstrings, which are documentation, never count.
    Calls of `import_module` or `__import__` with a constant module name count as imports.
    """
    package = module if is_package else module.rpartition('.')[0]
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split('.')
                base = base[: len(base) - (node.level - 1)]
                target = '.'.join([*base, *([node.module] if node.module else [])])
            else:
                target = node.module or ''
            found += [target, *(f'{target}.{alias.name}' for alias in node.names)]
        elif isinstance(node, ast.Call):
            function = node.func
            name = function.attr if isinstance(function, ast.Attribute) else getattr(function, 'id', None)
            argument = node.args[0] if node.args else None
            if name in DYNAMIC_IMPORTS and isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                found.append(argument.value)
    return found


def _imports_models(source, module, *, is_package=False):
    """Return the modules of `skcomplex.models` a core source imports."""
    imported = _imported_modules(source, module, is_package=is_package)
    return sorted({target for target in imported if target == MODELS or target.startswith(f'{MODELS}.')})


def test_core_imports_no_models():
    """Test that no module of the core imports `skcomplex.models`, directly, relatively, or by name."""
    found = []
    for path in sorted(CORE.rglob('*.py')):
        module, is_package = _module_of(path)
        found += [
            f'{module}: imports {target}' for target in _imports_models(path.read_text(), module, is_package=is_package)
        ]
    assert found == []


@pytest.mark.parametrize(
    'source',
    [
        'import skcomplex.models\n',
        'import skcomplex.models.physics as physics\n',
        'from skcomplex.models import create\n',
        'from skcomplex import models\n',
        'from ...models import physics\n',
        'from ... import models\n',
        'from ...models.physics.realizations import Spin\n',
        "from importlib import import_module\nimport_module('skcomplex.models.physics')\n",
        "import importlib\nimportlib.import_module('skcomplex.models')\n",
        "__import__('skcomplex.models')\n",
        'def f():\n    from ...models import run\n',
    ],
)
def test_import_check_finds_planted_imports(source):
    """Test that the import check finds direct, relative, aliased, nested, and dynamic imports of the models."""
    assert _imports_models(source, 'skcomplex.core.model._planted')


@pytest.mark.parametrize(
    'source',
    [
        '"""Run a model.\n\nExamples:\n    >>> from skcomplex.models import run\n"""\n',
        'from ..model import System\nfrom ...core.base import Domain\n',
        "MODULE = 'skcomplex.models'\n",
    ],
)
def test_import_check_allows_doctests_and_core_imports(source):
    """Test that doctests in docstrings, imports of the core, and plain strings are not imports of the models."""
    assert _imports_models(source, 'skcomplex.core.model._planted') == []


def test_core_public_names_have_no_field_terms():
    """Test that no name in the `__all__` of a package of the core contains a field term."""
    found = []
    for path in sorted(CORE.rglob('__init__.py')):
        module = _module_of(path)[0]
        found += [f'{module}.{name}: {terms}' for name in _public_names(path) if (terms := _field_terms(name))]
    assert found == []


def test_core_modules_have_no_field_terms():
    """Test that no package or module file of the core contains a field term."""
    paths = [path for path in CORE.rglob('*') if path.suffix == '.py' or (path / '__init__.py').exists()]
    found = [
        f'{path.relative_to(CORE)}: {terms}'
        for path in sorted(paths)
        if (terms := _field_terms(path.stem if path.suffix == '.py' else path.name))
    ]
    assert found == []


@pytest.mark.parametrize(
    ('name', 'terms'),
    [
        ('Market', ['market']),
        ('markets', ['market']),
        ('LimitOrderBook', ['order book']),
        ('order_books', ['order book']),
        ('LennardJones', ['lennard']),
        ('SpinChain', ['spin']),
        ('QUBIT_SPECIES', ['species', 'qubit']),
        ('_predator_prey', ['predator', 'prey']),
        ('Turtle', ['turtle']),
        ('ElementaryWolfram', ['wolfram']),
        ('LangevinThermostat', ['thermostat']),
        ('spinodal', []),
        ('preys_on', ['prey']),
        ('Generator', []),
        ('StockFlow', []),
        ('PairPotential', []),
        ('OrderParameter', []),
        ('WordGeometry', []),
    ],
)
def test_field_terms_match_whole_words(name, terms):
    """Test that field terms match whole words of snake case and CamelCase names, with plurals, and nothing inside."""
    assert sorted(_field_terms(name)) == sorted(terms)
