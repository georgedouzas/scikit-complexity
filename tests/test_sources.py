"""Test the sources of the repository: the package, its tests, and its documentation hide no finding of the checks."""

from pathlib import Path

import pytest

FOLDERS = ('src', 'tests', 'docs')
GENERATED = Path('docs') / 'generated'
FORBIDDEN = (
    'TYPE' + '_CHECKING',
    'no' + 'qa',
    'type' + ': ignore',
    'no' + 'sec',
)


def _sources(root: Path) -> list[Path]:
    """Return the Python files of the package, its tests, and its documentation, without the generated pages."""
    return [
        path
        for folder in FOLDERS
        for path in sorted((root / folder).rglob('*.py'))
        if not path.relative_to(root).is_relative_to(GENERATED)
    ]


@pytest.mark.parametrize('word', FORBIDDEN)
def test_no_source_hides_a_finding(request, word):
    """Test that no source imports for type checking only or suppresses a finding of a linter or a type checker."""
    root = request.config.rootpath
    found = [
        f'{path.relative_to(root)}:{number}'
        for path in _sources(root)
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1)
        if word in line
    ]
    assert not found, f'{word} appears in: {", ".join(found)}'
