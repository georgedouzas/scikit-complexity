"""Generate the compatibility page from the declarations of the laws, solvers, backends, and formalisms."""

import pkgutil
from collections.abc import Iterable
from importlib import import_module

from skcomplex.core import formalisms
from skcomplex.core.base import LawKind
from skcomplex.core.laws import law_classes
from skcomplex.core.space import BaseSpace
from skcomplex.engine.compile import BACKENDS, SOLVERS
from skcomplex.formalisms import registered_formalisms


def _import_formalisms() -> None:
    """Import every formalism, so its laws, spaces, solvers, and backends are registered."""
    for module in pkgutil.iter_modules(formalisms.__path__):
        if module.ispkg:
            import_module(f'{formalisms.__name__}.{module.name}')


def _names(items: Iterable[object]) -> str:
    """Return the names of classes, terms, or keys in declaration order, `none` for no item."""
    names = [getattr(item, '__name__', str(item)) for item in items]
    return ', '.join(f'`{name}`' for name in names) or 'none'


def _terms(terms: frozenset[object], every: int | None = None) -> str:
    """Return a set of terms as a sorted list, `any` for an empty requirement or a set of every term."""
    if not terms or (every is not None and len(terms) == every):
        return 'any'
    return ', '.join(f'`{term}`' for term in sorted(map(str, terms)))


def _noise_terms(terms: frozenset[object]) -> str:
    """Return the noise structures of a declaration, `none` for an empty one."""
    return _terms(terms) if terms else 'none'


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
    for law in law_classes():
        requires = law.requires
        lines.append(
            f'| `{law.__name__}` | `{law.law_kind}` | {law.time_class} | {_terms(requires.time)} | '
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
        'components, space, and noise structures of a system, and lists the solvers that support what it lacks. A',
        'diagonal noise draws one Wiener process per state, a shared noise drives several states with one process, and',
        'a correlated noise draws the processes of a Wiener element with a covariance, and a conserved noise draws one',
        'process per link of the lattice of a field.',
        '',
        '| Solver | Form | Time | Space | Laws | Domains | Components | Noise |',
        '|---|---|---|---|---|---|---|---|',
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
            _terms(supports.laws, len(LawKind.registered())),
            _terms(supports.domains, 12),
            _terms(supports.components, 7),
            _noise_terms(supports.noise),
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
        'Backends turn the compiled laws into numeric functions. Each declares the laws, domains, and noise structures',
        'it lowers.',
        '',
        '| Backend | Laws | Domains | Noise |',
        '|---|---|---|---|',
    ]
    for key in sorted(BACKENDS):
        supports = BACKENDS.resolve(key).supports
        cells = [
            _terms(supports.laws, len(LawKind.registered())),
            _terms(supports.domains, 12),
            _noise_terms(supports.noise),
        ]
        lines.append(f'| `{key}` | {" | ".join(cells)} |')
    return lines


def _spaces(base: type[BaseSpace]) -> list[type[BaseSpace]]:
    """Return every subclass of a space class that declares its kind, recursively."""
    found: list[type[BaseSpace]] = []
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
    for space in sorted(_spaces(BaseSpace), key=lambda item: item.__name__):
        lines.append(
            f'| `{space.__name__}` | `{space.space_kind}` | {_terms(space.lacks) if space.lacks else "none"} |',
        )
    return lines


def formalism_lines() -> list[str]:
    """Return the contributions of every registered formalism.

    Returns:
        lines:
            The Markdown lines of the section.
    """
    lines = [
        '',
        '## Formalisms',
        '',
        'Every formalism registers its contributions when it is imported, and the tables above list them with those of',
        'the shared core. A formalism registers after the formalisms it depends on.',
        '',
        '| Formalism | Depends on | Laws | Components | Forms | Passes | Solvers | Backends | Visuals |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for formalism in registered_formalisms():
        cells = [
            f'`{formalism.name}`',
            _names(formalism.depends),
            _names(formalism.laws),
            _names(formalism.components),
            _names(formalism.forms),
            _names(type(item) for item in formalism.passes),
            _names(formalism.solvers),
            _names(formalism.backends),
            _names(formalism.visuals),
        ]
        lines.append(f'| {" | ".join(cells)} |')
    return lines


def compatibility_page() -> str:
    """Return the compatibility page of laws, spaces, solvers, backends, and formalisms, from their declarations."""
    _import_formalisms()
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
        *formalism_lines(),
    ]
    return '\n'.join(lines) + '\n'


def main() -> None:
    """Write the compatibility page into the documentation."""
    import mkdocs_gen_files  # noqa: PLC0415

    with mkdocs_gen_files.open('overview/compatibility.md', 'w') as file:
        file.write(compatibility_page())


main()
