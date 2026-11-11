"""Development tasks."""

import argparse
import os
import shutil
import tomllib
from pathlib import Path

import nox

os.environ.update({'PDM_IGNORE_SAVED_PYTHON': '1'})


PYTHON_VERSIONS: list[str] = ['3.12', '3.13', '3.14']
# mkdocs-gallery reads `ast.Str`, which Python 3.14 removed.
DOCS_PYTHON: str = '3.13'
FILES: list[str] = ['src', 'tests', 'docs', 'noxfile.py']
CONVENTIONS: str = 'openspec/scripts/check_conventions.py'
EXTRAS: list[str] = ['-G', 'structures', '-G', 'rna', '-G', 'metabolism', '-G', 'relativity', '-G', 'trajectories']
QUICK_TESTS: list[str] = [
    'tests/core/components/elements',
    'tests/core/components/spaces',
    'tests/core/engine/backends',
    'tests/core/engine/compile',
    'tests/core/engine/problems',
    'tests/core/engine/runtime',
    'tests/core/engine/solvers/test_lammps.py',
    'tests/core/engine/solvers/test_scipy_ode.py',
    'tests/core/engine/viz',
    'tests/core/experiments',
    'tests/core/specs',
    'tests/core/test_base.py',
    'tests/core/test_errors.py',
    'tests/core/test_specs.py',
    'tests/core/test_types.py',
]
TEST_GROUPS: list[list[str]] = [
    ['--slow', 'tests/systems'],
    ['--slow', 'tests', '--ignore=tests/systems'],
    ['src/skcomplex', 'README.md', 'docs/overview/user_guide'],
]
CHANGELOG_ARGS: list[str] = [
    '--convention',
    'conventional',
    '--template',
    'keepachangelog',
    '--trailers',
    '--sections',
    'feat,fix,docs,style,refactor,tests,chore',
    '--bump',
    'auto',
    '.',
]


def check_cli(session: nox.Session, args: list[str], default: str = 'all') -> str:
    """Check the CLI arguments.

    Arguments:
        session: The nox session.
        args: The available CLI arguments.
        default: The subcommand used when none is given.
    """
    available_args = ', '.join([f'`{arg}`' for arg in args])
    msg = f'Available subcommands are one of {available_args}.'
    session_args = list(session.posargs) or [default]
    if len(session_args) > 1 or session_args[0] not in args:
        session.skip(f'{msg} Instead `{" ".join(session_args)}` was given')
    return session_args[0]


@nox.session
def clean(session: nox.Session) -> None:
    """Clean build artifacts and cache files.

    Arguments:
        session: The nox session.
    """
    paths = [
        '.mypy_cache',
        '.pytest_cache',
        'tests/.pytest_cache',
        'build',
        'dist',
        'htmlcov',
        'pip-wheel-metadata',
        'site',
        '__pycache__',
        'docs/generated',
        '.nox',
        '.ruff_cache',
        'coverage.xml',
    ]
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)
    for cache_path in Path().rglob('__pycache__'):
        shutil.rmtree(cache_path, ignore_errors=True)
    for file in Path().rglob('*.rej'):
        if file.exists():
            file.unlink()
    for file in Path().rglob('.coverage*'):
        if file.exists():
            file.unlink()


@nox.session(python=DOCS_PYTHON)
def docs(session: nox.Session) -> None:
    """Build or serve the documentation.

    Arguments:
        session: The nox session.
    """
    arg = check_cli(session, ['serve', 'build'], default='serve')
    session.run('pdm', 'install', '-dG', 'docs', *EXTRAS, external=True)
    # mkdocs-gallery imports MkDocs' deprecated `warning_filter` when the plugin loads, before any hook can filter it.
    env = {'PYTHONWARNINGS': "ignore:warning_filter doesn't do anything:DeprecationWarning", 'PYTHONPATH': 'docs'}
    session.run('properdocs', arg, env=env)


@nox.session(python=DOCS_PYTHON)
def gallery(session: nox.Session) -> None:
    """Run the gallery examples, capture their screenshots into the cache, and build the documentation with them.

    Arguments:
        session: The nox session.
    """
    parser = argparse.ArgumentParser(prog='nox -s gallery --')
    parser.add_argument('--backend', choices=['native', 'web'], default='native')
    parser.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument('--sample', action='store_true', help='run three fast systems per field only')
    arguments = parser.parse_args(session.posargs)
    session.run('pdm', 'install', '-dG', 'docs', '-dG', 'gallery', *EXTRAS, external=True)
    if arguments.backend == 'web':
        session.run('pdm', 'run', 'python', '-m', 'playwright', 'install', 'chromium', external=True)
    selection = [] if arguments.sample else ['--all']
    env = {
        'PYTHONPATH': 'docs',
        'SKCOMPLEX_GALLERY': '' if arguments.sample else 'all',
        'SKCOMPLEX_GALLERY_BACKEND': arguments.backend,
    }
    run = ['pdm', 'run', 'python', '-m', 'gallery_tools', 'run', '--backend', arguments.backend]
    session.run(*run, '--jobs', str(arguments.jobs), *selection, env=env, external=True)
    session.run('pdm', 'run', 'properdocs', 'build', '--strict', env=env, external=True)


@nox.session(python=PYTHON_VERSIONS[-1])
@nox.parametrize('file', FILES)
def formatting(session: nox.Session, file: str) -> None:
    """Format the code.

    Arguments:
        session: The nox session.
        file: The file to be formatted.
    """
    arg = check_cli(session, ['all', 'code', 'docstrings'])
    session.run('pdm', 'install', '-dG', 'formatting', '--no-default', external=True)
    # Black runs last, since docformatter sets the blank lines around some docstrings differently.
    if arg in ['docstrings', 'all']:
        session.run('docformatter', file, success_codes=[0, 3])
    if arg in ['code', 'all']:
        session.run('black', file)


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize('file', FILES)
def checks(session: nox.Session, file: str) -> None:
    """Check code quality, dependencies, type annotations, or the conventions.

    Arguments:
        session: The nox session.
        file: The file to be checked.
    """
    arg = check_cli(session, ['all', 'quality', 'dependencies', 'types', 'security', 'docs', 'conventions'])
    session.run('pdm', 'install', '-dG', 'checks', '--no-default', external=True)
    if arg in ['quality', 'all']:
        session.run('ruff', 'check', file)
        if file == 'src':
            session.run('lint-imports', env={'PYTHONPATH': 'src'})
    if arg in ['types', 'all']:
        session.run('mypy', file)
    if arg in ['security', 'all']:
        session.run('bandit', '-c', 'pyproject.toml', '-r', file, '-x', 'docs/gallery_tools')
        if file == 'docs':
            # The gallery tools run the installed Rerun viewer in a subprocess.
            session.run('bandit', '-c', 'pyproject.toml', '-r', 'docs/gallery_tools', '-s', 'B101,B404,B603')
    if arg in ['docs', 'all'] and file == 'src':
        session.run('interrogate', file)
        session.run('pydoclint', file)
    if arg in ['dependencies', 'all']:
        requirements_path = (Path(session.create_tmp()) / 'requirements.txt').as_posix()
        # The runtime dependencies are the default group, so only the development groups leave it out.
        args_groups = [['--prod']] + [['-dG', group, '--no-default'] for group in ['tests', 'docs', 'maintenance']]
        requirements_types = zip(FILES, args_groups, strict=True)
        args = [
            'pdm',
            'export',
            '-f',
            'requirements',
            '--without-hashes',
            '--pyproject',
            '-o',
            requirements_path,
        ]
        session.run(*(args + dict(requirements_types)[file]), external=True)
        config = tomllib.loads(Path('pyproject.toml').read_text())
        ignored = config.get('tool', {}).get('pip-audit', {}).get('ignore-vulns', [])
        ignore_args = [flag for vuln in ignored for flag in ('--ignore-vuln', vuln)]
        session.run('pip-audit', '-r', requirements_path, *ignore_args)
        session.run('deptry', 'src', 'tests')
    if arg in ['conventions', 'all'] and file == 'src' and session.python == PYTHON_VERSIONS[0]:
        session.run('ruff', 'check', CONVENTIONS)
        session.run('mypy', CONVENTIONS)
        # The checks import the package, so they run in the project environment, which keeps its runtime dependencies
        # away from the type checker of this one.
        env = {'PDM_IGNORE_ACTIVE_VENV': '1'}
        session.run('pdm', 'install', '--prod', *EXTRAS, env=env, external=True)
        session.run('pdm', 'run', 'python', CONVENTIONS, env=env, external=True)


@nox.session(python=PYTHON_VERSIONS)
def quick_tests(session: nox.Session) -> None:
    """Run a small subset of the fast tests, which CI runs on every push in a few minutes.

    Arguments:
        session: The nox session.
    """
    session.run('pdm', 'install', '-dG', 'tests', *EXTRAS, external=True)
    session.run('pytest', '-n', 'auto', *QUICK_TESTS)


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run tests and coverage.

    Arguments:
        session: The nox session.
    """
    session.run('pdm', 'install', '-dG', 'tests', *EXTRAS, external=True)
    env = {'COVERAGE_FILE': f'.coverage.{session.python}'}
    options = ['-n', 'auto', '--cov=src', '--cov=tests', '--cov-append']
    if session.posargs:
        session.run('pytest', *options, '--slow', '-k', *session.posargs, env=env)
    else:
        # Each group runs in its own workers, since the compiled code of a worker stays in its memory until it exits.
        for group in TEST_GROUPS:
            session.run('pytest', *options, *group, env=env)
    session.run('coverage', 'combine')
    session.run('coverage', 'report')
    session.run('coverage', 'html')
    session.run('coverage', 'xml')


@nox.session(python=PYTHON_VERSIONS[-1])
def changelog(session: nox.Session) -> None:
    """Build the changelog.

    Arguments:
        session: The nox session.
    """
    session.run('pdm', 'install', '-dG', 'changelog', '--no-default', external=True)
    session.run('git-changelog', *CHANGELOG_ARGS, '--output', 'CHANGELOG.md')


@nox.session(python=PYTHON_VERSIONS[-1])
def release(session: nox.Session) -> None:
    """Kick off a release process.

    Arguments:
        session: The nox session.
    """
    session.run('pdm', 'install', '-dG', 'changelog', '--no-default', external=True)
    session.run('git-changelog', *CHANGELOG_ARGS, '--output', 'CHANGELOG.md')
    version = str(session.run('git-changelog', *CHANGELOG_ARGS, '--bumped-version', silent=True)).strip()

    # Create release branch and commit changelog
    session.run('git', 'checkout', '-b', f'release_{version}', external=True)
    session.run('git', 'add', 'CHANGELOG.md', external=True)
    session.run('git', 'commit', '-m', f'chore: Release {version}', '--allow-empty', external=True)
    session.run('git', 'push', '-u', 'origin', f'release_{version}', external=True)

    # Create and merge PR from release branch to main
    session.run('gh', 'pr', 'create', '--base', 'main', '--fill', external=True)
    session.run('gh', 'pr', 'checks', '--watch', '--fail-fast', external=True)
    session.run('gh', 'pr', 'merge', '--merge', '--delete-branch', external=True)

    # Create and push the release tag, which is what the pipeline reacts to
    session.run('git', 'checkout', 'main', external=True)
    session.run('git', 'pull', '--rebase', external=True)
    session.run('git', 'tag', version, external=True)
    session.run('git', 'push', 'origin', version, external=True)
