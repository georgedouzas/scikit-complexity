"""Generate the compatibility page from the declarations of the laws, solvers, and backends."""

from collections.abc import Collection, Iterable

import mkdocs_gen_files

from skcomplex.core import laws
from skcomplex.core.laws import BaseLaw
from skcomplex.engine import solvers
from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.solvers import BaseSolver
from skcomplex.specs import SupportSpec

BACKENDS = (NumpyBackend, NumbaBackend)


def _name(item: object) -> str:
    """Return the name of a class, `no space` for `None`, or the text of a term."""
    if item is None:
        return 'no space'
    return str(getattr(item, '__name__', item))


def _terms(terms: Iterable[object]) -> str:
    """Return a set of terms as a sorted list, `any` for an empty declaration, which supports every term."""
    names = sorted(_name(term) for term in terms)
    return ', '.join(f'`{name}`' for name in names) or 'any'


def _law_classes() -> list[type[BaseLaw]]:
    """Return the public law classes of the core."""
    found: list[type[BaseLaw]] = []
    for name in laws.__all__:
        item = getattr(laws, name)
        if isinstance(item, type) and issubclass(item, BaseLaw) and item is not BaseLaw:
            found.append(item)
    return found


def _solver_classes() -> list[type[BaseSolver]]:
    """Return the public solver classes that declare their form and support, which their abstract bases do not."""
    found: list[type[BaseSolver]] = []
    for name in solvers.__all__:
        item = getattr(solvers, name)
        if isinstance(item, type) and issubclass(item, BaseSolver) and _declares(item):
            found.append(item)
    return sorted(found, key=lambda solver: (str(solver._get_form()), solver.__name__))


def _declares(solver: type[BaseSolver]) -> bool:
    """Check whether a solver class declares the form it solves and what it supports."""
    try:
        solver._get_form()
        solver._get_support()
    except NotImplementedError:
        return False
    return True


def _supporting(law: type[BaseLaw], supports: Collection[tuple[str, SupportSpec]]) -> str:
    """Return the names that support a law class, `none` for no name."""
    names = [name for name, support in supports if not support.missing('laws', [law])]
    return ', '.join(f'`{name}`' for name in names) or 'none'


def law_lines() -> list[str]:
    """Return the solvers and backends that support every law class.

    Returns:
        lines:
            The Markdown lines of the section.
    """
    lines = [
        '## Laws',
        '',
        'Every law states what it requires of a system, and every solver and backend declares the laws it supports.',
        'Validation reports a law that the chosen solver or backend does not support.',
        '',
        '| Law | Solvers | Backends |',
        '|---|---|---|',
    ]
    solver_supports = [(solver.__name__, solver._get_support()) for solver in _solver_classes()]
    backend_supports = [(backend.__name__, backend._get_support()) for backend in BACKENDS]
    for law in _law_classes():
        lines.append(
            f'| `{law.__name__}` | {_supporting(law, solver_supports)} | {_supporting(law, backend_supports)} |',
        )
    return lines


def solver_lines() -> list[str]:
    """Return the support of every solver.

    Returns:
        lines:
            The Markdown lines of the section.
    """
    lines = [
        '',
        '## Solvers',
        '',
        'Every solver declares what it supports. Validation checks a chosen solver against the time, space, laws,',
        'noise structures, and derivative orders of a system, and names what it lacks. A diagonal noise draws one',
        'Wiener process per state, a shared noise drives several states with one process, a correlated noise draws the',
        'processes of a `WienerProcessSpec` with a covariance, and a conserved noise draws one process per link of the',
        'lattice of a field.',
        '',
        '| Solver | Form | Time | Space | Laws | Noise | Orders |',
        '|---|---|---|---|---|---|---|',
    ]
    for solver in _solver_classes():
        supports = solver._get_support()
        cells = [
            f'`{solver.__name__}`',
            str(solver._get_form()),
            _terms(supports.time),
            _terms(supports.space),
            _terms(supports.laws),
            _terms(supports.noise),
            _terms(supports.orders),
        ]
        lines.append(f'| {" | ".join(cells)} |')
    return lines


def backend_lines() -> list[str]:
    """Return the support of every backend.

    Returns:
        lines:
            The Markdown lines of the section.
    """
    lines = [
        '',
        '## Backends',
        '',
        'Backends turn the compiled laws into numeric functions. Each declares the laws, noise structures, and',
        'derivative orders it lowers.',
        '',
        '| Backend | Laws | Noise | Orders |',
        '|---|---|---|---|',
    ]
    for backend in BACKENDS:
        supports = backend._get_support()
        cells = [_terms(supports.laws), _terms(supports.noise), _terms(supports.orders)]
        lines.append(f'| `{backend.__name__}` | {" | ".join(cells)} |')
    return lines


def compatibility_page() -> str:
    """Return the compatibility page of laws, solvers, and backends, from their declarations.

    Returns:
        page:
            The Markdown page.
    """
    lines = [
        '# Compatibility',
        '',
        'Not every combination of laws, spaces, time, solvers, and backends is valid. This page is generated from the',
        'declarations that validation reads, so it always agrees with what the compilers accept.',
        '',
        *law_lines(),
        *solver_lines(),
        *backend_lines(),
    ]
    return '\n'.join(lines) + '\n'


def main() -> None:
    """Write the compatibility page into the documentation."""
    with mkdocs_gen_files.open('overview/compatibility.md', 'w') as file:
        file.write(compatibility_page())


main()
