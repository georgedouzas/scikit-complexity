"""Test the layout of the package: the core never depends on models, and model packages are self-contained.

A model package imports the core and the shared realizations of its own field, and nothing of another package.
"""

import ast
import importlib.util
import subprocess  # nosec B404: the tests run the interpreter on code they write
import sys
from pathlib import Path

import pytest

from .layout_checks import (
    definition_order,
    formalism_imports,
    guarded_imports,
    leaking_exports,
    misordered_definitions,
    undeclared_formalism_imports,
    upward_imports,
    wide_base_imports,
)

PACKAGE = Path(__file__).parent.parent / 'src' / 'skcomplex'
MODELLING_PACKAGES = ['base', 'space', 'laws', 'components', 'interactions', 'system']
ENGINEERING_PACKAGES = ['compile', 'solvers', 'runtime', 'viz', 'experiments', 'analysis', 'backends']
CORE_PACKAGES = [*MODELLING_PACKAGES, *ENGINEERING_PACKAGES, 'formalisms']
FORMALISMS = 'skcomplex.core.formalisms'


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


def _core_violations(module, target):
    """Return the violations of an import of a core module: a model, or a formalism outside the formalisms."""
    if not module.startswith('skcomplex.core'):
        return []
    if target.startswith('skcomplex.models'):
        return [f'{module}: core imports {target}']
    if target.startswith(FORMALISMS) and not module.startswith(FORMALISMS):
        return [f'{module}: the shared core imports the formalism {target}']
    return []


def _violations(path):
    """Return the layout violations of one source file."""
    module, is_package = _module_of(path)
    found = []
    for target, names in _imports(path):
        if not target.startswith('skcomplex'):
            continue
        found += _core_violations(module, target)
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
            'core/compile/_planted_check.py',
            'from ..formalisms.lattices import LocalRule\n',
            'compile._planted_check: the shared core imports the formalism skcomplex.core.formalisms.lattices',
        ),
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
    """Test that the layout check reports a reverse, a formalism, a private, and a cross-package import."""
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
    """Test that a system built from scratch compiles, runs, and is analysed without the models or a formalism."""
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
            "assert not any(name.startswith('skcomplex.core.formalisms') for name in sys.modules), 'formalisms'",
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


ROOT = PACKAGE.parent
TESTS = Path(__file__).parent
DOCS = TESTS.parent / 'docs'
LAYOUT_TESTS = frozenset({Path(__file__).name, 'layout_checks.py'})
RESTRUCTURE = pytest.mark.xfail(strict=False, reason='enabled by task 5.2')


def _plant(root, files):
    """Write a small package under a root, with an `__init__` in every directory a file needs, and return the root."""
    for relative, source in files.items():
        path = root / relative
        for parent in reversed(path.relative_to(root).parents[:-1]):
            (root / parent).mkdir(exist_ok=True)
            (root / parent / '__init__.py').touch()
        path.write_text(source)
    return root


def _clients():
    """Return the tests and the documentation that use the exports of the core, leaving out the layout tests."""
    tests = [path for path in TESTS.rglob('*') if path.is_file() and path.name not in LAYOUT_TESTS]
    return [*tests, DOCS]


def test_upward_import_is_found(tmp_path):
    """Test that an import of a package listed later names the module and the package it reached."""
    root = _plant(
        tmp_path,
        {
            'skcomplex/core/solvers/_planted.py': 'from ..backends import Numba\nfrom ..compile import Problem\n',
            'skcomplex/core/laws/_planted.py': 'from skcomplex.core import system\n',
        },
    )
    assert upward_imports(root) == [
        'skcomplex.core.laws._planted: laws imports skcomplex.core.system, which comes after it',
        'skcomplex.core.solvers._planted: solvers imports skcomplex.core.backends, which comes after it',
    ]


@RESTRUCTURE
def test_core_imports_only_earlier_packages():
    """Test that every core package imports only the packages listed before it."""
    assert upward_imports(ROOT) == []


@pytest.mark.parametrize(
    ('source', 'expected'),
    [
        (
            'class Public:\n    pass\n\n\ndef _helper():\n    pass\n',
            ['planted: the private function _helper comes after the public class Public'],
        ),
        (
            'def public():\n    pass\n\n\nclass _Private:\n    pass\n\n\ndef _helper():\n    pass\n',
            ['planted: the private function _helper comes after the private class _Private'],
        ),
        ('def _a():\n    pass\n\n\ndef b():\n    pass\n\n\nclass _C:\n    pass\n\n\nclass D:\n    pass\n', []),
        ('class Base:\n    pass\n\n\nclass _Derived(Base):\n    pass\n', []),
        ('def decorate(f):\n    return f\n\n\n@decorate\ndef _decorated():\n    pass\n', []),
        ('def default():\n    pass\n\n\ndef _uses(f=default):\n    pass\n', []),
        ('class Value:\n    pass\n\n\nVALUE = Value()\n\n\ndef _after():\n    pass\n', []),
        (
            (
                'def _inner():\n    pass\n\n\ndef outer():\n    return _inner()\n\n\nSHAPE = outer()\n\n\n'
                'class _Shape:\n    size = SHAPE\n'
            ),
            [],
        ),
        (
            'class Base:\n    pass\n\n\nclass _Derived(Base):\n    pass\n\n\ndef _late():\n    pass\n',
            ['planted: the private function _late comes after the private class _Derived'],
        ),
    ],
)
def test_helper_between_classes_is_found(source, expected):
    """Test that a definition after one of a later turn is named, unless the module needs it before its turn."""
    assert misordered_definitions(source, 'planted') == expected


def test_definition_order_names_the_module(tmp_path):
    """Test that the order check of the tree names the module of a misplaced helper."""
    root = _plant(tmp_path, {'skcomplex/core/laws/_planted.py': 'class Law:\n    pass\n\n\ndef _helper():\n    pass\n'})
    expected = 'skcomplex.core.laws._planted: the private function _helper comes after the public class Law'
    assert definition_order(root) == [expected]


@RESTRUCTURE
def test_core_modules_define_in_order():
    """Test that every core module defines private functions, public functions, private classes, then public ones."""
    assert definition_order(ROOT) == []


@pytest.mark.parametrize(
    ('relative', 'source', 'expected'),
    [
        (
            'skcomplex/core/components/_component.py',
            'from ..space import Space\nfrom ..base import Domain\nimport numpy\n\n\nclass Component:\n    pass\n',
            [
                (
                    'skcomplex.core.components._component: defines the base Component and imports '
                    'skcomplex.core.space, not only skcomplex.core.base'
                ),
            ],
        ),
        (
            'skcomplex/core/laws/_law.py',
            'from ._values import values_of\n\n\nclass Law:\n    pass\n',
            [
                (
                    'skcomplex.core.laws._law: defines the base Law and imports skcomplex.core.laws._values, '
                    'not only skcomplex.core.base'
                ),
            ],
        ),
        ('skcomplex/core/laws/_law.py', 'from ..base import Domain\n\n\nclass Law:\n    pass\n', []),
    ],
)
def test_base_class_with_wide_imports_is_found(tmp_path, relative, source, expected):
    """Test that a module defining `Component` or `Law` that imports beyond `skcomplex.core.base` is named."""
    assert wide_base_imports(_plant(tmp_path, {relative: source})) == expected


def test_base_classes_import_only_base():
    """Test that the modules defining `Component` and `Law` import only `skcomplex.core.base`."""
    assert wide_base_imports(ROOT) == []


def test_guarded_import_is_found(tmp_path):
    """Test that an import under `TYPE_CHECKING`, bare or as an attribute, names the module and the import."""
    source = (
        'import typing\nfrom typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from ..space import Space\n'
        '\nif typing.TYPE_CHECKING:\n    import numpy\n'
    )
    root = _plant(tmp_path, {'skcomplex/core/system/_planted.py': source})
    assert guarded_imports(root) == [
        'skcomplex.core.system._planted:5: imports skcomplex.core.space under TYPE_CHECKING',
        'skcomplex.core.system._planted:8: imports numpy under TYPE_CHECKING',
    ]


@RESTRUCTURE
def test_core_guards_no_import():
    """Test that no core module imports under `TYPE_CHECKING`."""
    assert guarded_imports(ROOT) == []


def test_planted_formalism_import_is_found(tmp_path):
    """Test that a shared core module importing a formalism is named, and a formalism importing the core is not."""
    root = _plant(
        tmp_path,
        {
            'skcomplex/core/compile/_planted.py': 'from ..formalisms.lattices import LocalRule\n',
            'skcomplex/core/solvers/_planted.py': 'from skcomplex.core import formalisms\n',
            'skcomplex/core/formalisms/lattices/_laws.py': 'from ...laws import Law\n',
        },
    )
    assert formalism_imports(root) == [
        'skcomplex.core.compile._planted: the shared core imports skcomplex.core.formalisms.lattices',
        'skcomplex.core.solvers._planted: the shared core imports skcomplex.core.formalisms',
    ]


def test_shared_core_imports_no_formalism():
    """Test that no module of the shared core imports from `skcomplex.core.formalisms`."""
    assert formalism_imports(ROOT) == []


def test_undeclared_formalism_import_is_found(tmp_path):
    """Test that a formalism importing one it does not declare names both, and a declared one passes."""
    root = _plant(
        tmp_path,
        {
            'skcomplex/core/formalisms/agents/__init__.py': "from .._formalism import Formalism\nFormalism('agents')\n",
            'skcomplex/core/formalisms/lattices/__init__.py': "FORMALISM = Formalism(name='lattices')\n",
            'skcomplex/core/formalisms/fields/__init__.py': (
                "from .. import Formalism, lattices\nFORMALISM = Formalism('fields', depends=('lattices',))\n"
            ),
            'skcomplex/core/formalisms/kinetics/__init__.py': "FORMALISM = Formalism('kinetics')\n",
            'skcomplex/core/formalisms/kinetics/_laws.py': (
                'from ..agents import Rule\nfrom ...formalisms import lattices\nfrom .._formalism import register\n'
            ),
        },
    )
    prefix = 'skcomplex.core.formalisms.kinetics._laws: the formalism kinetics imports the formalism'
    assert undeclared_formalism_imports(root) == [
        f'{prefix} agents, which it does not declare',
        f'{prefix} lattices, which it does not declare',
    ]


def test_no_formalisms_means_no_undeclared_import(tmp_path):
    """Test that a tree without formalisms has no undeclared formalism import."""
    assert undeclared_formalism_imports(_plant(tmp_path, {'skcomplex/core/laws/_law.py': ''})) == []


def test_formalisms_import_only_declared_formalisms():
    """Test that every formalism imports only the formalisms its declaration lists in `depends`."""
    assert undeclared_formalism_imports(ROOT) == []


def test_compiler_internal_leak_is_found(tmp_path):
    """Test that an export one other core package alone uses is named, unless a client or a second package uses it."""
    root = _plant(
        tmp_path,
        {
            'skcomplex/core/laws/__init__.py': (
                'from ._laws import Rate, shift, request_of, unused\n'
                "__all__ = ['Rate', 'shift', 'request_of', 'unused']\n"
            ),
            'skcomplex/core/compile/_planted.py': 'from ..laws import Rate, shift, request_of\n',
            'skcomplex/core/solvers/_planted.py': 'from ..laws._laws import shift\n',
            'skcomplex/models/physics/_planted.py': 'from skcomplex.core.laws import Rate\n',
        },
    )
    expected = 'skcomplex.core.laws: exports request_of, which only skcomplex.core.compile uses'
    assert leaking_exports(root, []) == [expected]
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs' / 'guide.md').write_text('Read the request with `request_of(law)`.\n')
    assert leaking_exports(root, [tmp_path / 'docs']) == []
    assert leaking_exports(root, [], concepts=['skcomplex.core.laws.request_of']) == []


@RESTRUCTURE
def test_core_exports_do_not_leak():
    """Test that no core package exports a name that one other core package alone uses."""
    assert leaking_exports(ROOT, _clients()) == []
