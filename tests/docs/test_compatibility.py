"""Test the compatibility page generated from the declarations of laws, solvers, and backends."""

from dataclasses import replace
from pathlib import Path

from skcomplex.core.base import Form, LawKind
from skcomplex.core.solvers import SOLVERS

SCRIPT = Path(__file__).parents[2] / 'docs' / 'generate_compatibility.py'


def _generator():
    """Return the namespace of the compatibility generator, without writing the page."""
    namespace = {'__name__': 'generate_compatibility', '__file__': str(SCRIPT)}
    source = SCRIPT.read_text().removesuffix('\nmain()\n')
    exec(compile(source, str(SCRIPT), 'exec'), namespace)  # noqa: S102  # nosec B102: runs the repository's own script
    return namespace


def _row(page, start):
    """Return the row of the page that starts with a cell."""
    return next(line for line in page.splitlines() if line.startswith(start))


def test_page_lists_every_law_and_solver():
    """Test that the page has a row for every law with its kind and for every named solver."""
    page = _generator()['compatibility_page']()
    assert '| `Rate` | `rate` | continuous | `continuous` |' in page
    assert '`monte-carlo`' in page
    for key in SOLVERS:
        if key[1] != 'auto':
            assert f'| `{key[1]}` |' in page, key


def test_page_lists_the_spaces_and_what_they_lack():
    """Test that a group space is a lattice-kind space that lacks a regular geometry and an image layout."""
    page = _generator()['compatibility_page']()
    assert _row(page, '| `GroupSpace` |') == "| `GroupSpace` | `lattice` | `image`, `regular-geometry` |"
    assert _row(page, '| `LatticeSpace` |') == '| `LatticeSpace` | `lattice` | none |'


def test_table_follows_declarations(monkeypatch):
    """Test that a solver whose declared support gains a law kind is listed with it at the next generation."""
    stepper = SOLVERS.resolve((Form.DISCRETE, 'iteration'))
    assert '`noise`' not in _row(_generator()['compatibility_page'](), '| `iteration` |')
    monkeypatch.setattr(stepper, 'supports', replace(stepper.supports, laws=stepper.supports.laws | {LawKind.NOISE}))
    assert '`noise`' in _row(_generator()['compatibility_page'](), '| `iteration` |')
