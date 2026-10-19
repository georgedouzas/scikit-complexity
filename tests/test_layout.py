"""Test the layout of the package: the core never depends on models, and model packages are self-contained."""

import ast
import subprocess  # nosec B404: the tests run the interpreter on code they write
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parent.parent / 'src' / 'skcomplex'
CORE_PACKAGES = ['base', 'model', 'compile', 'solvers', 'runtime', 'viz', 'experiments', 'analysis', 'backends']


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
            aggregator = is_package and target.startswith(f'{module}.')
            own = _home(target) == _home(module) or target.startswith(f'{_home(module)}.')
            if not aggregator and not own and target != _home(module):
                found.append(f'{module}: imports the model package {target}')
    return found


def test_layout_has_core_and_models():
    """Test that the package holds exactly the core and the models, and the core holds every framework package."""
    children = sorted(path.name for path in PACKAGE.iterdir() if path.is_dir() and path.name != '__pycache__')
    assert children == ['core', 'models']
    core = sorted(path.name for path in (PACKAGE / 'core').iterdir() if path.is_dir() and path.name != '__pycache__')
    assert core == sorted(CORE_PACKAGES)


def test_framework_imports_from_core():
    """Test that the framework's names import from the packages of the core."""
    from skcomplex.core.base import Domain  # noqa: PLC0415
    from skcomplex.core.model import Element  # noqa: PLC0415
    from skcomplex.core.runtime import Simulation  # noqa: PLC0415
    from skcomplex.core.solvers import RK4  # noqa: PLC0415

    assert all(item is not None for item in (Domain, Element, Simulation, RK4))


@pytest.mark.parametrize('name', ['skcomplex.model', 'skcomplex.solvers', 'skcomplex.runtime'])
def test_old_framework_paths_are_gone(name):
    """Test that the framework packages are no longer importable from their old paths."""
    with pytest.raises(ModuleNotFoundError):
        __import__(name)


def test_package_respects_layout():
    """Test that no source file of the package breaks the layout rules."""
    found = [item for path in PACKAGE.rglob('*.py') for item in _violations(path)]
    assert found == []


@pytest.mark.parametrize(
    ('relative', 'source', 'message'),
    [
        ('core/model/_planted_check.py', 'from ...models import physics\n', 'core imports skcomplex.models'),
        (
            'models/_planted/_check.py',
            'from ...core.compile._compiler import _compile\n',
            'private _compiler, _compile',
        ),
        ('models/_planted/_check.py', 'from ..physics import mechanics\n', 'imports the model package'),
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
            'from skcomplex.core.model import Element, Rate, System',
            'from skcomplex.core.runtime import Simulation',
            "Decay = Element.define('Decay', states={'n': 1.0}, laws=lambda s: [Rate(s.n, -s.n)])",
            "system = System(elements=[Decay('a')])",
            'Simulation(system, dt=0.1).run(until=1)',
            'equations_of(system)',
            "assert not any(name.startswith('skcomplex.models') for name in sys.modules), 'models imported'",
        ],
    )
    subprocess.run([sys.executable, '-c', code], check=True)  # noqa: S603  # nosec B603: runs code written above
