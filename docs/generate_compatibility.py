"""Generate the compatibility page from the declarations of the elements, interactions, laws, solvers, and backends."""

import inspect
from collections.abc import Collection, Iterable
from types import ModuleType

import mkdocs_gen_files

from skcomplex.core.components import elements, interactions, laws
from skcomplex.core.components.elements import BaseElement, GroupElement
from skcomplex.core.components.interactions import BaseInteraction
from skcomplex.core.components.laws import BaseLaw
from skcomplex.core.engine import solvers
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.solvers import BaseSolver
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec, SupportSpec
from skcomplex.core.types import ElementKind

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


def _public_classes[T](package: ModuleType, base: type[T]) -> list[type[T]]:
    """Return the public classes of a package that derive from a base, the base included."""
    named = [getattr(package, name) for name in package.__all__]
    return [item for item in named if isinstance(item, type) and issubclass(item, base)]


def _element_kinds(element: type[BaseElement]) -> str:
    """Return the kinds of element that a class of elements fills a role as."""
    if not issubclass(element, GroupElement):
        return f'`{ElementKind.ELEMENT}`'
    kinds = [kind for kind in ElementKind if kind not in {ElementKind.ANY, ElementKind.ELEMENT, ElementKind.SYSTEM}]
    return ', '.join(f'`{kind}`' for kind in kinds)


def _names_of(element: type[BaseElement], spec: type) -> str:
    """Return the names an element class declares with a spec, `none` for no name."""
    names = [name for name, item in element._get_specs().items() if isinstance(item, spec)]
    return ', '.join(f'`{name}`' for name in names) or 'none'


def _accepted(role: RoleSpec) -> str:
    """Return the kinds of element a role accepts, or what else it binds."""
    if role.noise:
        return 'a `WienerProcessSpec`'
    if not role.elements or ElementKind.ANY in role.elements:
        return 'any'
    return ', '.join(f'`{kind}`' for kind in role.elements)


def _read(interaction: type[BaseInteraction], name: str) -> str:
    """Return a declared read of a role: the argument that names the state, with the default of the argument."""
    default = inspect.signature(interaction).parameters[name].default
    if isinstance(default, str):
        return f'`{name}` (default `{default}`)'
    return f'`{name}`'


def _role_cell(interaction: type[BaseInteraction], name: str, role: RoleSpec) -> str:
    """Return one role of an interaction: its name, the kinds it accepts, and the arguments that name its states."""
    reads = ', '.join(_read(interaction, item) for item in role.states)
    optional = ', optional' if role.optional else ''
    many = ', many' if role.variadic else ''
    return f'`{name}`{optional}{many}: accepts {_accepted(role)}; reads {reads or "nothing it declares"}'


def _interaction_roles(interaction: type[BaseInteraction]) -> str:
    """Return the roles of an interaction, one line of a table cell per role."""
    roles = [(name, item) for name, item in interaction._get_specs().items() if isinstance(item, RoleSpec)]
    return '<br>'.join(_role_cell(interaction, name, role) for name, role in roles) or 'a subclass declares them'


def _law_cells(interaction: type[BaseInteraction]) -> list[str]:
    """Return the laws of an interaction with their time and space, or the note that only running it shows them."""
    classes = interaction._get_law_classes()
    if not classes:
        return ['known only by running a subclass', 'by its laws', 'by its laws']
    needs = [law.get_requirements() for law in classes]
    return [
        ', '.join(f'`{law.__name__}`' for law in classes),
        _terms(BaseInteraction._get_accepted([need.time for need in needs])),
        _terms(BaseInteraction._get_accepted([need.space for need in needs])),
    ]


def element_lines() -> list[str]:
    """Return the kinds of element and the states and parameters of every element class.

    Returns:
        lines:
            The Markdown lines of the section.
    """
    lines = [
        '## Elements',
        '',
        'An element fills a role of an interaction. `get_states()` of an element returns the states it carries, and',
        'the classes below declare only these. Your own subclass adds the states your interactions read.',
        '',
        '| Element | Kinds | States | Parameters |',
        '|---|---|---|---|',
    ]
    for element in _public_classes(elements, BaseElement):
        cells = [_element_kinds(element), _names_of(element, StateSpec), _names_of(element, ParameterSpec)]
        lines.append(f'| `{element.__name__}` | {" | ".join(cells)} |')
    return lines


def interaction_lines() -> list[str]:
    """Return the roles, the laws, and the time and space of every interaction class.

    Returns:
        lines:
            The Markdown lines of the section.
    """
    lines = [
        '',
        '## Interactions',
        '',
        'Every role names the kinds of element it accepts and the states it reads. Most roles read a state that an',
        'argument names, so `RateInteraction(state=\'n\')` reads `n` from its `target`. `get_requirements()` of an',
        'interaction returns the values for the arguments you pass. A function you pass can read other states, and',
        'validation reports a missing one as `role-element` when it builds the laws. A `time` or `space` of `any`',
        'means that one of the laws runs in any.',
        '',
        '| Interaction | Roles | Laws | Time | Space |',
        '|---|---|---|---|---|',
    ]
    for interaction in _public_classes(interactions, BaseInteraction):
        cells = [_interaction_roles(interaction), *_law_cells(interaction)]
        lines.append(f'| `{interaction.__name__}` | {" | ".join(cells)} |')
    return lines


def law_lines() -> list[str]:
    """Return the solvers and backends that support every law class.

    Returns:
        lines:
            The Markdown lines of the section.
    """
    lines = [
        '',
        '## Laws',
        '',
        'Every law states what it requires of a system, and every solver and backend declares the laws it supports.',
        'Validation reports a law that the chosen solver or backend does not support.',
        '',
        '| Law | Time | Space | Solvers | Backends |',
        '|---|---|---|---|---|',
    ]
    solver_supports = [(solver.__name__, solver._get_support()) for solver in _solver_classes()]
    backend_supports = [(backend.__name__, backend._get_support()) for backend in BACKENDS]
    for law in _law_classes():
        needs = law.get_requirements()
        cells = [
            _terms(needs.time),
            _terms(needs.space),
            _supporting(law, solver_supports),
            _supporting(law, backend_supports),
        ]
        lines.append(f'| `{law.__name__}` | {" | ".join(cells)} |')
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
    """Return the compatibility page of elements, interactions, laws, solvers, and backends, from their declarations.

    Returns:
        page:
            The Markdown page.
    """
    lines = [
        '# Compatibility',
        '',
        'Not every combination of elements, interactions, laws, spaces, time, solvers, and backends is valid. This',
        'page is generated from the declarations that validation reads, so it agrees with what the compilers accept.',
        'Choose in this order: time and space, then elements, then interactions, then the solver. The section',
        '[Choosing compatible components](user_guide.md#choosing-compatible-components) of the user guide walks',
        'through one example.',
        '',
        *element_lines(),
        *interaction_lines(),
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
