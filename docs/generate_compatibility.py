"""Generate the compatibility page from the requirements of the laws and the support of the solvers and backends."""

from skcomplex.core.base import LawKind
from skcomplex.core.compile import BACKENDS, law_types
from skcomplex.core.solvers import SOLVERS
from skcomplex.core.space import Space


def _terms(terms: frozenset[object], every: int | None = None) -> str:
    """Return a set of terms as a sorted list, `any` for an empty requirement or a set of every term."""
    if not terms or (every is not None and len(terms) == every):
        return 'any'
    return ', '.join(f'`{term}`' for term in sorted(map(str, terms)))


def law_lines() -> list[str]:
    """Return the requirements of every law type.

    Returns:
        lines:
            The Markdown lines of the section.
    """
    lines = [
        '## Laws',
        '',
        'Every law declares its kind, the time semantics of the kind, and what it requires of the system: the time it',
        'runs in, the space it lives on, and the domains of the states it targets. An empty requirement accepts any.',
        '',
        '| Law | Kind | Class | Time | Space | Domains |',
        '|---|---|---|---|---|---|',
    ]
    for law in law_types():
        requires = law.requires
        lines.append(
            f'| `{law.__name__}` | `{law.law_kind}` | {law.law_kind.law_class} | {_terms(requires.time)} | '
            f'{_terms(requires.space)} | {_terms(requires.domains)} |',
        )
    return lines


def solver_lines() -> list[str]:
    """Return the support of every registered solver.

    Returns:
        lines:
            The Markdown lines of the section.
    """
    lines = [
        '',
        '## Solvers',
        '',
        'Every solver declares what it supports. Validation checks a chosen solver against the laws, domains,',
        'components, and space of a system, and lists the solvers that support what it lacks.',
        '',
        '| Solver | Form | Time | Space | Laws | Domains | Components |',
        '|---|---|---|---|---|---|---|',
    ]
    for key in sorted(SOLVERS, key=lambda key: (key[0].name, key[1])):
        if key[1] == 'auto':
            continue
        supports = SOLVERS.resolve(key).supports
        cells = [
            f'`{key[1]}`',
            key[0].name,
            _terms(supports.time),
            _terms(supports.space, 5),
            _terms(supports.laws, len(LawKind)),
            _terms(supports.domains, 12),
            _terms(supports.components, 7),
        ]
        lines.append(f'| {" | ".join(cells)} |')
    return lines


def backend_lines() -> list[str]:
    """Return the support of every registered backend.

    Returns:
        lines:
            The Markdown lines of the section.
    """
    lines = [
        '',
        '## Backends',
        '',
        'Backends turn the compiled laws into numeric functions. Each declares the laws and domains it lowers.',
        '',
        '| Backend | Laws | Domains |',
        '|---|---|---|',
    ]
    for key in sorted(BACKENDS):
        supports = BACKENDS.resolve(key).supports
        lines.append(f'| `{key}` | {_terms(supports.laws, len(LawKind))} | {_terms(supports.domains, 12)} |')
    return lines


def _spaces(base: type) -> list[type]:
    """Return every subclass of a space class that declares its kind, recursively."""
    found = []
    for subclass in base.__subclasses__():
        found += [subclass, *_spaces(subclass)]
    return [space for space in found if 'space_kind' in vars(space) or 'lacks' in vars(space)]


def space_lines() -> list[str]:
    """Return the kind of every space and the options of its kind that it lacks.

    Returns:
        lines:
            The Markdown lines of the section.
    """
    lines = [
        '',
        '## Spaces',
        '',
        'Every space declares its kind, which the requirements of laws and the support of solvers name, and the',
        'options of its kind it lacks. Validation reports a law or a visual that needs an option its space lacks, such',
        'as the checkerboard schedule, which needs the regular geometry of a square lattice.',
        '',
        '| Space | Kind | Lacks |',
        '|---|---|---|',
    ]
    for space in sorted(_spaces(Space), key=lambda item: item.__name__):
        lines.append(
            f'| `{space.__name__}` | `{space.space_kind}` | {_terms(space.lacks) if space.lacks else "none"} |',
        )
    return lines


def compatibility_page() -> str:
    """Return the compatibility page of laws, solvers, and backends, generated from their declarations."""
    lines = [
        '# Compatibility',
        '',
        'Not every combination of laws, spaces, time, solvers, and backends is valid. This page is generated from the',
        'declarations that validation reads, so it always agrees with what `validate` accepts.',
        '',
        *law_lines(),
        *space_lines(),
        *solver_lines(),
        *backend_lines(),
    ]
    return '\n'.join(lines) + '\n'


def main() -> None:
    """Write the compatibility page into the documentation."""
    import mkdocs_gen_files  # noqa: PLC0415

    with mkdocs_gen_files.open('overview/compatibility.md', 'w') as file:
        file.write(compatibility_page())


main()
