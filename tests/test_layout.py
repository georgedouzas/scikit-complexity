"""Test the layout of the package: the core never depends on models, and model packages are self-contained.

A model package imports the core and the shared realizations of its own field, and nothing of another package.
"""

import ast
import importlib.util
import subprocess  # nosec B404: the tests run the interpreter on code they write
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parent.parent / 'src' / 'skcomplex'
MODELLING_PACKAGES = ['base', 'space', 'laws', 'components', 'interactions', 'system']
CORE_PACKAGES = [*MODELLING_PACKAGES, 'compile', 'solvers', 'runtime', 'viz', 'experiments', 'analysis', 'backends']


def _module_of(path):
    """Return the dotted module of a source file and whether it is a package's `__init__`."""
    parts = list(path.relative_to(PACKAGE.parent).with_suffix('').parts)
    if parts[-1] == '__init__':
        return '.'.join(parts[:-1]), True
    return '.'.join(parts), False


def _imports(path):
    """Return the absolute modules a source file imports, each with the names it imports from them."""
    module, is_package = _module_of(path)
    package = module if is_package else module.rpartition('.')[0]
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found += [(alias.name, []) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split('.')
                base = base[: len(base) - (node.level - 1)]
                target = '.'.join([*base, *([node.module] if node.module else [])])
            else:
                target = node.module or ''
            found.append((target, [alias.name for alias in node.names]))
    return found


def _home(module):
    """Return the model package a module belongs to: its field and subfield, or its field."""
    parts = module.split('.')
    return '.'.join(parts[:4]) if len(parts) > 4 else '.'.join(parts[:3])


def _violations(path):
    """Return the layout violations of one source file."""
    module, is_package = _module_of(path)
    found = []
    for target, names in _imports(path):
        if not target.startswith('skcomplex'):
            continue
        if module.startswith('skcomplex.core') and target.startswith('skcomplex.models'):
            found.append(f'{module}: core imports {target}')
        if not module.startswith('skcomplex.models'):
            continue
        if target.startswith('skcomplex.core'):
            private = [part for part in target.split('.')[2:] if part.startswith('_')]
            private += [name for name in names if name.startswith('_')]
            if private:
                found.append(f'{module}: private {", ".join(private)} from {target}')
        elif target.startswith('skcomplex.models'):
            field = '.'.join(module.split('.')[:3])
            if target.startswith(f'{field}.realizations'):
                continue
            aggregator = is_package and target.startswith(f'{module}.')
            own = _home(target) == _home(module) or target.startswith(f'{_home(module)}.')
            if not aggregator and not own and target != _home(module):
                found.append(f'{module}: imports the model package {target}')
    return found


GROUP_READS = frozenset({'agents', 'count', 'create'})


def _roles_of(node):
    """Return the names of the roles a class body declares with `role()`."""
    return {
        item.target.id
        for item in node.body
        if isinstance(item, ast.AnnAssign)
        and isinstance(item.target, ast.Name)
        and isinstance(item.value, ast.Call)
        and getattr(item.value.func, 'id', None) == 'role'
    }


def _is_own_label(node):
    """Check whether a formatted string starts with the component's own label followed by a dot."""
    values = node.values
    return (
        len(values) > 1
        and isinstance(values[0], ast.FormattedValue)
        and ast.unparse(values[0].value) == 'self.label'
        and isinstance(values[1], ast.Constant)
        and str(values[1].value).startswith('.')
    )


def _group_read(node, roles):
    """Return the group a call of the model view names, when it is neither a role nor a role's path."""
    function = node.func
    if not (isinstance(function, ast.Attribute) and function.attr in GROUP_READS and node.args):
        return None
    if ast.unparse(function.value) != 'model':
        return None
    argument = node.args[0]
    if isinstance(argument, ast.Constant) and argument.value not in roles:
        return repr(argument.value)
    is_attribute = isinstance(argument, ast.Attribute) and ast.unparse(argument.value) == 'self'
    if is_attribute and argument.attr not in roles:
        return f'self.{argument.attr}'
    return None


def _rule_violations(source, module):
    """Return the rules of interactions that read groups other than through roles, or parameters by their label."""
    found = []
    tree = ast.parse(source)
    found += [
        f'{module}:{node.lineno}: reads a parameter by its own label'
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr) and _is_own_label(node)
    ]
    for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        roles = _roles_of(cls)
        if not roles and 'Interaction' not in {ast.unparse(base) for base in cls.bases}:
            continue
        for node in ast.walk(cls):
            group = _group_read(node, roles) if isinstance(node, ast.Call) else None
            if group is not None:
                found.append(f'{module}:{node.lineno}: {cls.name} reads the group {group}, which is not a role')
    return found


def test_layout_has_core_and_models():
    """Test that the package holds exactly the core and the models, and the core holds every framework package."""
    children = sorted(path.name for path in PACKAGE.iterdir() if path.is_dir() and path.name != '__pycache__')
    assert children == ['core', 'models']
    core = sorted(path.name for path in (PACKAGE / 'core').iterdir() if path.is_dir() and path.name != '__pycache__')
    assert core == sorted(CORE_PACKAGES)


def _top_level_imports(path):
    """Return the core packages a module imports when it loads, leaving out imports under `TYPE_CHECKING`."""
    module, is_package = _module_of(path)
    package = module if is_package else module.rpartition('.')[0]
    found = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.level:
            base = package.split('.')[: len(package.split('.')) - (node.level - 1)]
            target = '.'.join([*base, *([node.module] if node.module else [])])
        elif isinstance(node, ast.ImportFrom):
            target = node.module or ''
        else:
            continue
        parts = target.split('.')
        if parts[:2] == ['skcomplex', 'core'] and len(parts) > 2:
            found.add(parts[2])
    return found


def test_modelling_packages_import_downward():
    """Test that each modelling package of the core imports only the packages below it, as a system is defined."""
    found = []
    for rank, name in enumerate(MODELLING_PACKAGES):
        allowed = set(MODELLING_PACKAGES[: rank + 1])
        for path in sorted((PACKAGE / 'core' / name).rglob('*.py')):
            found += [f'{path.name} of {name} imports {target}' for target in _top_level_imports(path) - allowed]
    assert found == []


def test_primitives_import_from_their_part_of_a_system():
    """Test that primitives import from the package of their part: system, components, interactions, space, laws."""
    from skcomplex.core.components import Element  # noqa: PLC0415
    from skcomplex.core.interactions import Interaction  # noqa: PLC0415
    from skcomplex.core.laws import Rate  # noqa: PLC0415
    from skcomplex.core.space import DiscreteTime, LatticeSpace  # noqa: PLC0415
    from skcomplex.core.system import System  # noqa: PLC0415

    assert importlib.util.find_spec('skcomplex.core.model') is None
    modules = {
        item.__module__.split('.')[2] for item in (Element, Interaction, Rate, DiscreteTime, LatticeSpace, System)
    }
    assert modules == {'components', 'interactions', 'laws', 'space', 'system'}


def test_framework_imports_from_core():
    """Test that the framework's names import from the packages of the core."""
    from skcomplex.core.base import Domain  # noqa: PLC0415
    from skcomplex.core.components import Element  # noqa: PLC0415
    from skcomplex.core.runtime import Simulation  # noqa: PLC0415
    from skcomplex.core.solvers import RK4  # noqa: PLC0415

    assert all(item is not None for item in (Domain, Element, Simulation, RK4))


@pytest.mark.parametrize('name', ['skcomplex.model', 'skcomplex.solvers', 'skcomplex.runtime'])
def test_old_framework_paths_are_gone(name):
    """Test that the framework packages are no longer importable from their old paths."""
    with pytest.raises(ModuleNotFoundError):
        __import__(name)


def test_realizations_of_the_own_field_import():
    """Test that a model package imports the realizations of its own field, and not those of another field."""
    path = PACKAGE / 'models' / 'physics' / 'mechanics' / '_planted_realizations.py'
    path.write_text('from ..realizations import Spin\nfrom ...economics.realizations import Market\n')
    try:
        found = _violations(path)
    finally:
        path.unlink()
    expected = 'skcomplex.models.physics.mechanics._planted_realizations: imports the model package '
    assert found == [f'{expected}skcomplex.models.economics.realizations']


def test_package_respects_layout():
    """Test that no source file of the package breaks the layout rules."""
    found = [item for path in PACKAGE.rglob('*.py') for item in _violations(path)]
    assert found == []


@pytest.mark.parametrize(
    ('relative', 'source', 'message'),
    [
        ('core/system/_planted_check.py', 'from ...models import physics\n', 'core imports skcomplex.models'),
        (
            'models/_planted/_check.py',
            'from ...core.compile._compiler import _compile\n',
            'private _compiler, _compile',
        ),
        ('models/_planted/_check.py', 'from ..physics import mechanics\n', 'imports the model package'),
        ('models/_planted/_check.py', 'from ..physics.realizations import Spin\n', 'imports the model package'),
    ],
)
def test_layout_check_finds_planted_imports(relative, source, message):
    """Test that the layout check reports a reverse, a private, and a cross-package import."""
    path = PACKAGE / relative
    created = not path.parent.exists()
    path.parent.mkdir(exist_ok=True)
    path.write_text(source)
    try:
        assert any(message in item for item in _violations(path))
    finally:
        path.unlink()
        if created:
            path.parent.rmdir()


def test_core_works_without_models():
    """Test that a system built from scratch compiles, runs, and is analysed without importing the model library."""
    code = '\n'.join(
        [
            'import sys',
            'from skcomplex.core.analysis import equations_of',
            'from skcomplex.core.components import Element',
            'from skcomplex.core.laws import Rate',
            'from skcomplex.core.system import System',
            'from skcomplex.core.runtime import Simulation',
            "Decay = Element.define('Decay', states={'n': 1.0}, laws=lambda s: [Rate(s.n, -s.n)])",
            "system = System(components=[Decay('a')])",
            'Simulation(system, dt=0.1).run(until=1)',
            'equations_of(system)',
            "assert not any(name.startswith('skcomplex.models') for name in sys.modules), 'models imported'",
        ],
    )
    subprocess.run([sys.executable, '-c', code], check=True)  # noqa: S603  # nosec B603: runs code written above


def test_no_rule_reads_its_own_label():
    """Test that rules read their parameters through the scoped view and groups of interactions through roles."""
    paths = (PACKAGE / 'models').rglob('*.py')
    found = [item for path in paths for item in _rule_violations(path.read_text(), _module_of(path)[0])]
    assert found == []


@pytest.mark.parametrize(
    ('source', 'message'),
    [
        ("def f(self, model):\n    return model.parameters[f'{self.label}.rate']\n", 'by its own label'),
        (
            "class Herding:\n    cows: str = role()\n\n    def herd(self, model):\n        model.agents('pasture')\n",
            "reads the group 'pasture'",
        ),
        (
            (
                "class Day:\n    women: str = role()\n    premises: str = 'premises'\n\n    def run(self, model):\n"
                '        model.count(self.premises)\n'
            ),
            'reads the group self.premises',
        ),
    ],
)
def test_rule_check_finds_planted_reads(source, message):
    """Test that the rule check reports reads by the own label and by group paths that are not roles."""
    assert any(message in item for item in _rule_violations(source, 'planted'))


def test_backends_live_in_one_package():
    """Test that every registered backend, and the Mesa stepper, is defined in a private module of `core.backends`."""
    from skcomplex.core.backends import MesaStepper  # noqa: PLC0415
    from skcomplex.core.compile import BACKENDS, Backend  # noqa: PLC0415

    backends = [BACKENDS.resolve(name) for name in BACKENDS]
    modules = {item.__module__ for item in [*backends, MesaStepper]}
    assert {'skcomplex.core.backends._numpy', 'skcomplex.core.backends._numba'} <= modules
    assert all(module.startswith('skcomplex.core.backends._') for module in modules)
    defined = [
        f'{path.relative_to(PACKAGE)}: {node.name}'
        for package in ('compile', 'solvers')
        for path in sorted((PACKAGE / 'core' / package).rglob('*.py'))
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and any(ast.unparse(base) == 'Backend' for base in node.bases)
    ]
    assert defined == []
    assert Backend.__module__ == 'skcomplex.core.compile._backends'
