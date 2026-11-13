"""Check the conventions of the repository: the package, its tests, and its documentation hide no finding.

Run it from the root of the repository, in the environment of the project, with `pdm run python
openspec/scripts/check_conventions.py`. It prints every finding as `file:line: message` and exits with 1 when it finds
one. It first checks its own readers on planted files, so a reader that stops finding what it should is a finding too.
"""

import ast
import dataclasses
import doctest
import enum
import importlib
import inspect
import multiprocessing
import pkgutil
import re
import sys
import tempfile
import textwrap
import tokenize
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, cast, is_typeddict

SCRIPT = Path('openspec') / 'scripts' / 'check_conventions.py'
FOLDERS = ('src', 'tests', 'docs')
SOURCE = Path('src') / 'skcomplex'
TESTS = Path('tests')
README = 'README.md'
USER_GUIDE = Path('docs') / 'overview' / 'user_guide'
PYTHON_BLOCK = re.compile(r'^```python[^\n]*\n(.*?)^```$', re.MULTILINE | re.DOTALL)
SYSTEMS = 'skcomplex.systems'
CATALOG_FUNCTIONS = ('all_systems',)
GENERAL = f'{SYSTEMS}.general'
SUBFIELD_PACKAGE = re.compile(rf'{re.escape(SYSTEMS)}\.[a-z]\w*\.[a-z]\w*')
FAMILY_PACKAGE = re.compile(rf'{re.escape(SYSTEMS)}\.[a-z]\w*\.common\.[a-z]\w*')
SYSTEMS_PATH = re.compile(rf'{re.escape(SYSTEMS)}\.[\w<>.]*[\w>]')
SUBFIELD_PLACEHOLDER = f'{SYSTEMS}.<field>.<subfield>'
CONCEPT_PLACEHOLDER = f'{GENERAL}.<concept>'
GENERATED = Path('docs') / 'generated'
FORBIDDEN = (
    'TYPE' + '_CHECKING',
    'no' + 'qa',
    'type' + ': ignore',
    'no' + 'sec',
)
CONSTANT_NAME = re.compile(r'_*[A-Z][A-Z0-9_]*')
IMPORTS = (ast.Import, ast.ImportFrom)
PROPERTIES = ('property', 'cached_property')
PACKAGE = 'skcomplex'
DUNDER = re.compile(r'__\w+__')
LIBRARY_HOOK = re.compile(r'_[a-z][a-z0-9_]*[a-z0-9]_')
FACTORY_KEYWORD = re.compile(r'\w+_func|latex_name')
UTILS_MODULE = re.compile(r'\._utils$|\.utils\.')
FENCE = re.compile(r'^\s*(```|~~~)')
INLINE_CODE = re.compile('`[^`\n]*`')
CORE_COMPONENTS = USER_GUIDE / 'core_components'
FAMILY_PAGES = {
    'elements.md': 'skcomplex.core.components.elements',
    'interactions.md': 'skcomplex.core.components.interactions',
    'laws.md': 'skcomplex.core.components.laws',
    'spaces.md': 'skcomplex.core.components.spaces',
    'systems.md': 'skcomplex.core.components.systems',
}
GALLERY = Path('docs') / 'examples'
OWN_RECEIVERS = ('self', 'cls', 'super()')
DOCTESTS = doctest.DocTestParser()
CLASS_PACKAGES = (
    'skcomplex.core.components',
    'skcomplex.core.engine',
    'skcomplex.core.experiments',
    'skcomplex.core.analysis',
)
FIELD_TERMS = (
    'Ant',
    'Ask',
    'Bank',
    'Bid',
    'Bird',
    'Budget',
    'Climate',
    'Contagion',
    'Cow',
    'Credit',
    'Disease',
    'Energy',
    'Epidemic',
    'Firm',
    'Force',
    'Gene',
    'Grass',
    'Heading',
    'Infection',
    'Institution',
    'Market',
    'Neuron',
    'Opinion',
    'Pasture',
    'Predator',
    'Prey',
    'Price',
    'Protein',
    'Quantum',
    'Sheep',
    'Species',
    'Synapse',
    'Temperature',
    'Trade',
    'Traffic',
    'Voter',
    'Wolf',
    'Worker',
)
MOLECULAR_TERMS = ('rna', 'protein', 'genotype', 'hydrophobicity', 'residue', 'nucleotide')
ALLOWED_MAGIC_READS = frozenset(
    [
        '__init__',
        '__name__',
        '__qualname__',
        '__doc__',
        '__module__',
        '__mro__',
        '__version__',
        '__subclasses__',
    ],
)
SUMMARY_LENGTH = 80
SUFFIXES = (
    'Element',
    'Interaction',
    'Law',
    'Space',
    'Time',
    'System',
    'Solver',
    'Backend',
    'Observer',
    'Spec',
    'Result',
    'Pass',
    'Error',
)
KINDS = {suffix: suffix.lower() for suffix in SUFFIXES} | {'Neighborhood': 'neighbourhood'}
FAMILIES = (
    *(f'skcomplex.core.components.{family}' for family in ('elements', 'interactions', 'laws', 'spaces', 'systems')),
    'skcomplex.systems.general',
)
GENERATED_CLASSES_ALLOWED = {'core/engine/solvers/_mesa.py': 2}
FIELDS = (
    'biology',
    'chemistry',
    'computer_science',
    'economics',
    'engineering',
    'mathematics',
    'physics',
    'social',
)
CONNECTOR_PACKAGES = ('core/components', 'core/engine')
TYPE_ARGUMENTS = 3
SERVED_FIELDS = 2
PART_WORD = re.compile(r'(?<![\w`-])parts?(?![\w`-])', re.IGNORECASE)
NAME_WORD = re.compile(r'[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+')
PART_CONTEXT = 120
KEPT_PART_NAMES = frozenset(
    [
        'PARTS',
        'PART_NAMES',
        'Part',
        'PartElement',
        'PartView',
        'Parts',
        'PartsElement',
        'SECOND_PARTS',
        'TOPOLOGY_PARTS',
        '_LinearPartsSpec',
        '_compile_part',
        '_count_parts',
        '_get_delay_parts',
        '_get_dumbbell_parts',
        '_get_linear_parts',
        '_get_parts',
        '_parts',
        '_split_part',
        'antisymmetric_part',
        'delay_parts',
        'imag_part',
        'n_parts',
        'normal_part',
        'part',
        'parts',
        'real_part',
        'symmetric_part',
        'test_complex_component_is_located_by_its_part',
        'test_complex_equation_and_event_split_into_parts',
        'test_complex_functions_split_into_parts',
        'test_complex_ginzburg_landau_seeds_both_parts',
        'test_complex_value_of_real_state_names_its_imaginary_part',
        'test_group_on_parts_without_a_space_raises_error',
        'test_group_sizes_itself_from_the_parts_of_its_space',
        'test_parts_of_a_random_flow_are_orthogonal_and_sum_to_it',
        'test_point_fit_searches_the_central_part_of_a_scipy_prior',
        'test_prepare_rejects_an_inner_solver_that_lacks_its_part',
        'test_real_state_accepts_complex_value_with_zero_imaginary_part',
        'test_simulation_extend_runs_in_two_parts',
        'test_space_without_sites_has_no_parts',
        'test_split_solver_linear_part_matches_linear_rates',
        'test_symmetric_and_antisymmetric_parts_split_the_velocity_gradient',
        'test_unknown_part_is_rejected',
    ],
)
PART_QUALIFIERS = (
    r'(real|imaginary|complex|symmetric|antisymmetric|traceless|normal|linear|nonlinear|stiff|deterministic|random'
    r'|reversible|irreversible|divergence-free|gradient|curl|cyclic|harmonic|odd|even|positive|negative|integer'
    r'|fractional|constant|periodic|planetary|reciprocal|nonreciprocal|brownian|angular|exchange|transport|collision'
    r'|unique|shared|redundant|synergistic|independent|reader|other|training|validation|test|proliferating|irradiated'
    r'|outer|kept|first|last|adhesion|machine|field|general)'
)
KEPT_PART_PHRASES = tuple(
    re.compile(phrase, re.IGNORECASE)
    for phrase in (
        rf'\b{PART_QUALIFIERS}((,|\s)+(and|or|the|its|{PART_QUALIFIERS}))*\s+parts?\b',
        r'\b(the|has)\s+(two|three|four)\s+parts\b',
        r'\bthe\s+part\s+code\b',
        r'\bby\s+parts\b',
        r'\bin\s+part\b',
        r'\b(take|takes)\s+(no\s+)?part\b',
        r'\bplays?\s+(its\s+own\s+|no\s+)part\b',
        r'\b(is|are|be|not|no|itself|only|one)\s+part\s+(of|in)\b',
        r'\b(a\s+)?part\s+of\s+(it|them|the\s+(data|resource|gross\s+energy|library|market|swarm))\b',
        r'\bmore\s+than\s+its\s+parts\b',
        r'\bparts?:(\s|$)',
        r'\bspace\s+whose\s+parts\b',
        (
            r'\bparts?\s+of\s+(the\s+|a\s+|an\s+|its\s+(own\s+)?|every\s+|each\s+|that\s+|these\s+)?'
            r'(space|lattice|network|complex|cell|surface|face|frame)\b'
        ),
        r'\b(on|off)\s+(the\s+)?parts\b',
        r'\b(space\s+or|tied\s+to\s+no|have\s+the)\s+part\b',
        r'\bpart\s+other\s+than\s+the\s+sites\b',
        r'\bparts?\s+(its|whose)\s+group\b',
        r'\bnumber\s+of\s+(these\s+)?parts\b',
        (
            r'\bparts?\s+of\s+(the\s+|a\s+|an\s+|its\s+|every\s+|each\s+|their\s+)?(word|patch|structure|string|symbol\s+name'
            r'|identifier|path|dotted\s+label|value|matrix|tensor|vector|price|rates?|drift|amplitude|wave\s+function'
            r'|cross-diffusion|evolution|weighted\s+terms?|information|series|interval|program|problem|co-simulation'
            r'|step|trace|convolution|density|initial\s+density|sum|difference|relative\s+displacement|range'
            r'|compilation|engine|watch|technology)\b'
        ),
        r'\bparts\s+or\s+subassemblies\b',
        r'\bwatches\s+of\s+code\s+parts\b',
        r'\b[0-9]+\s+parts\b',
        r'(imaginary|modulus|absolute\s+values?|synerg\w*|redundan\w*)\W(.|\n){0,240}',
        r'(.|\n){0,240}\W(imaginary|modulus|absolute\s+values?|synerg\w*|redundan\w*)\b',
    )
)
BANNED_PHRASES = (
    'in summary',
    'in conclusion',
    'to summarize',
    'it is worth noting',
    "it's worth noting",
    'note that',
    "let's dive",
    'dive into',
    'delve',
    'leverage',
    'seamless',
    'seamlessly',
    'robust',
    'powerful',
    'cutting-edge',
    'unlock',
    'harness',
    'empower',
    'plethora',
    'myriad',
    'a wide range of',
    "in today's",
    'simply',
    'just',
    'easily',
    'magic',
)
VERBS = frozenset(
    [
        'accelerate',
        'accept',
        'accepts',
        'accumulate',
        'act',
        'activate',
        'add',
        'adjust',
        'adopt',
        'adopts',
        'advance',
        'align',
        'allocate',
        'append',
        'apply',
        'arrange',
        'arrive',
        'assemble',
        'assign',
        'attach',
        'average',
        'avoid',
        'balance',
        'barter',
        'bear',
        'begin',
        'bind',
        'binds',
        'borrow',
        'breaks',
        'breed',
        'broadcast',
        'build',
        'burn',
        'buy',
        'call',
        'carries',
        'carry',
        'catch',
        'change',
        'changes',
        'charge',
        'check',
        'choose',
        'claim',
        'clear',
        'climb',
        'close',
        'coerce',
        'collect',
        'collide',
        'combine',
        'combines',
        'commute',
        'compete',
        'compile',
        'compiles',
        'compute',
        'contains',
        'continues',
        'contract',
        'controls',
        'convert',
        'convolve',
        'copy',
        'count',
        'create',
        'cut',
        'decide',
        'declare',
        'declares',
        'decode',
        'defer',
        'deposit',
        'derive',
        'descend',
        'describe',
        'destine',
        'destroy',
        'die',
        'dies',
        'diffuse',
        'dispatch',
        'display',
        'disrupt',
        'divide',
        'draw',
        'drive',
        'drop',
        'duplicate',
        'embed',
        'emit',
        'encode',
        'enter',
        'enumerate',
        'evaluate',
        'evolve',
        'exchange',
        'excludes',
        'execute',
        'expand',
        'expect',
        'extend',
        'feed',
        'feel',
        'feels',
        'fight',
        'fill',
        'filter',
        'find',
        'finish',
        'fire',
        'fit',
        'fits',
        'fix',
        'flatten',
        'flip',
        'fly',
        'follow',
        'forecast',
        'forget',
        'fork',
        'format',
        'gain',
        'gather',
        'get',
        'give',
        'go',
        'grow',
        'handle',
        'happens',
        'harvest',
        'has',
        'herd',
        'hire',
        'hoist',
        'hold',
        'holds',
        'hop',
        'imitate',
        'increment',
        'infect',
        'infer',
        'influence',
        'initialize',
        'integrate',
        'intensify',
        'interact',
        'interpret',
        'invade',
        'invent',
        'is',
        'iterate',
        'join',
        'jump',
        'jumps',
        'kick',
        'land',
        'lays',
        'learn',
        'leave',
        'lend',
        'ligate',
        'link',
        'live',
        'load',
        'locate',
        'log',
        'lower',
        'make',
        'map',
        'mark',
        'mask',
        'match',
        'measure',
        'merge',
        'mirror',
        'move',
        'moves',
        'multiplies',
        'multiply',
        'mutate',
        'need',
        'negate',
        'nests',
        'note',
        'observe',
        'open',
        'operate',
        'pad',
        'pay',
        'parse',
        'partition',
        'pick',
        'place',
        'plan',
        'play',
        'pluck',
        'populate',
        'predict',
        'prepare',
        'print',
        'produce',
        'project',
        'proliferate',
        'propagate',
        'propose',
        'pull',
        'push',
        'quote',
        'raise',
        'rank',
        'react',
        'read',
        'reads',
        'rebuild',
        'record',
        'recount',
        'redesign',
        'redraw',
        'reduce',
        'regrow',
        'reject',
        'relax',
        'release',
        'remember',
        'remove',
        'rename',
        'renew',
        'replace',
        'replay',
        'replicate',
        'report',
        'reprice',
        'request',
        'reset',
        'resize',
        'resolve',
        'respond',
        'restart',
        'restock',
        'retarget',
        'returns',
        'rewire',
        'rewrite',
        'rotate',
        'run',
        'runs',
        'sample',
        'scale',
        'scatter',
        'schedule',
        'score',
        'scroll',
        'search',
        'searches',
        'secretes',
        'seed',
        'select',
        'sell',
        'separate',
        'set',
        'settle',
        'shift',
        'shop',
        'simulate',
        'smooth',
        'solve',
        'split',
        'spread',
        'spreads',
        'stalls',
        'start',
        'starves',
        'stay',
        'steer',
        'step',
        'stops',
        'strips',
        'substitute',
        'subtract',
        'sum',
        'sums',
        'supply',
        'support',
        'supports',
        'sweep',
        'swim',
        'swing',
        'switches',
        'tag',
        'take',
        'trade',
        'transfer',
        'translate',
        'transport',
        'travel',
        'triangulates',
        'truncate',
        'tune',
        'turn',
        'type',
        'unpack',
        'unroll',
        'update',
        'visit',
        'wait',
        'waits',
        'walk',
        'walks',
        'wander',
        'warm',
        'warn',
        'wire',
        'work',
        'wrap',
        'write',
    ],
)


def _locate(value: object, root: Path) -> str:
    """Return where a class or a function of the library is defined, as `file:line`, or its module file."""
    try:
        path = Path(str(inspect.getsourcefile(cast(type, value))))
        line = inspect.getsourcelines(cast(type, value))[1]
    except (TypeError, OSError):
        module = importlib.import_module(getattr(value, '__module__', None) or PACKAGE)
        return f'{_relative(Path(str(module.__file__)), root)}:1'
    return f'{_relative(path, root)}:{line}'


def _relative(path: Path, root: Path) -> Path:
    """Return a path relative to the root, or the path itself when it lies outside the root."""
    resolved = path.resolve()
    return resolved.relative_to(root.resolve()) if resolved.is_relative_to(root.resolve()) else path


def _sources(root: Path) -> list[Path]:
    """Return the Python files of the package, its tests, and its documentation, without the generated pages."""
    return [
        path
        for folder in FOLDERS
        for path in sorted((root / folder).rglob('*.py'))
        if not path.relative_to(root).is_relative_to(GENERATED)
    ]


def check_no_source_hides_a_finding(root: Path) -> list[str]:
    """Return every line that imports for type checking only or suppresses a finding of a linter or a type checker."""
    return [
        f'{path.relative_to(root)}:{number}: {word} appears'
        for word in FORBIDDEN
        for path in _sources(root)
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1)
        if word in line
    ]


def _documents(root: Path) -> list[Path]:
    """Return the README and the pages of the user guide, relative to the root."""
    return [Path(README), *sorted(path.relative_to(root) for path in (root / USER_GUIDE).rglob('*.md'))]


def _python_blocks(path: Path) -> list[tuple[int, str]]:
    """Return the Python blocks of a Markdown file, each with the line of the file before its first line."""
    text = path.read_text(encoding='utf-8')
    return [(text.count('\n', 0, match.start(1)), match.group(1)) for match in PYTHON_BLOCK.finditer(text)]


def _imported_names(code: str) -> list[tuple[int, str, str]]:
    """Return what a block of code imports, each as its line in the block, its module, and its name."""
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.ImportFrom) and node.module:
            found += [(node.lineno, node.module, alias.name) for alias in node.names]
        elif isinstance(node, ast.Import):
            found += [(node.lineno, alias.name, '') for alias in node.names]
    return found


def _is_allowed_import(module: str, name: str) -> bool:
    """Return whether an import takes a catalog function, a system class from its subfield, or a general realization.

    An analysis of one field and its result may also come from the family package of that field.
    """
    if module == SYSTEMS:
        return name in CATALOG_FUNCTIONS
    if module.startswith(f'{GENERAL}.'):
        return SUBFIELD_PACKAGE.fullmatch(module) is not None
    if FAMILY_PACKAGE.fullmatch(module) is not None:
        return name.islower() or name.endswith('Result')
    if module.startswith(f'{SYSTEMS}.'):
        return SUBFIELD_PACKAGE.fullmatch(module) is not None and name.endswith('System')
    return not (module == 'skcomplex' and name == 'systems')


def check_documents_import_systems_from_their_subfield(root: Path) -> list[str]:
    """Return every import of the README and the user guide beyond the system classes of a subfield."""
    return [
        f'{document}:{start + line}: imports {module} {name} beyond the system classes of a subfield'
        for document in _documents(root)
        for start, code in _python_blocks(root / document)
        for line, module, name in _imported_names(code)
        if not _is_allowed_import(module, name)
    ]


def _is_allowed_path(path: str) -> bool:
    """Return whether a dotted path below the systems names a subfield, a family of a field, or the general package."""
    if path in (SUBFIELD_PLACEHOLDER, GENERAL, CONCEPT_PLACEHOLDER):
        return True
    return SUBFIELD_PACKAGE.fullmatch(path) is not None or FAMILY_PACKAGE.fullmatch(path) is not None


def check_documents_name_only_subfield_packages(root: Path) -> list[str]:
    """Return every line of the README and the user guide that names a path below the systems but a subfield."""
    return [
        f'{document}:{number}: names a path below the systems other than a subfield: {line.strip()}'
        for document in _documents(root)
        for number, line in enumerate((root / document).read_text(encoding='utf-8').splitlines(), start=1)
        if not all(_is_allowed_path(path) for path in SYSTEMS_PATH.findall(line))
    ]


def check_readme_custom_system_runs_from_the_core_alone(root: Path) -> list[str]:
    """Return a finding when the first custom system of the README fails to simulate without the catalog."""
    text = (root / README).read_text(encoding='utf-8')
    start = text.index('### Build a custom system')
    match = PYTHON_BLOCK.search(text, start)
    if match is None:
        return [f'{README}:{text.count(chr(10), 0, start) + 1}: the custom system has no Python block']
    line = text.count('\n', 0, match.start(1))
    code = match.group(1)
    if 'simulate(' not in code:
        return [f'{README}:{line}: the custom system does not call `simulate`']
    check = f"""
import sys
{code}
loaded = [name for name in sys.modules if name == {SYSTEMS!r} or name.startswith({SYSTEMS + '.'!r})]
assert loaded == [], loaded
"""
    process = multiprocessing.get_context('spawn').Process(target=exec, args=(check, {'__name__': '__main__'}))
    process.start()
    process.join()
    if process.exitcode != 0:
        return [f'{README}:{line}: the custom system fails or loads the catalog in a fresh interpreter']
    return []


def _prose_faults(line: str) -> list[str]:
    """Return the semicolons, dashes used as punctuation, and banned phrases of a prose line without inline code."""
    faults = []
    if ';' in line:
        faults.append('semicolon')
    if re.search('[\u2013\u2014]', line) or ' - ' in line.strip()[1:]:
        faults.append('dash')
    lowered = line.lower()
    faults += [phrase for phrase in BANNED_PHRASES if re.search(rf'(?<![\w-]){re.escape(phrase)}(?![\w-])', lowered)]
    return faults


def _markdown_prose(text: str) -> list[tuple[int, str]]:
    """Return the prose lines of a Markdown text with their numbers, leaving out fenced blocks and inline code."""
    lines = []
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            lines.append((number, INLINE_CODE.sub('', line)))
    return lines


def _example_prose(text: str) -> list[tuple[int, str]]:
    """Return the prose lines of a gallery example with their numbers: its docstring and its comments."""
    docstring = ast.get_docstring(ast.parse(text), clean=False) or ''
    lines = [(number + 1, line) for number, line in _markdown_prose(docstring)]
    first = len(docstring.splitlines()) + 2
    for number, line in enumerate(text.splitlines()[first:], start=first + 1):
        if line.lstrip().startswith('#') and not line.lstrip().startswith('# %%'):
            lines.append((number, INLINE_CODE.sub('', line.lstrip()[1:])))
    return lines


def _unnatural_lines(root: Path, documents: tuple[str, ...]) -> list[str]:
    """Return every prose line of the documents with a semicolon, a dash, or a banned phrase, as `file:line: faults`."""
    found = []
    for document in documents:
        text = (root / document).read_text(encoding='utf-8')
        prose = _example_prose(text) if document.endswith('.py') else _markdown_prose(text)
        found += [
            f'{document}:{number}: {", ".join(faults)}: {line.strip()}'
            for number, line in prose
            if (faults := _prose_faults(line))
        ]
    return found


def check_documents_read_naturally(root: Path) -> list[str]:
    """Return every prose line of the README, the guide, and the gallery with a semicolon, dash, or banned phrase."""
    gallery = sorted(str(path.relative_to(root)) for path in (root / GALLERY).rglob('*.py'))
    documents = (*(str(path) for path in _documents(root)), 'CONTRIBUTING.md', *gallery)
    return _unnatural_lines(root, documents)


def check_core_components_pages_cover_every_class(root: Path) -> list[str]:
    """Return every public class without a section in its family page, and every import of the catalog there."""
    found = []
    for page, package in FAMILY_PAGES.items():
        path = root / CORE_COMPONENTS / page
        headings = set(re.findall(r'^## (\w+)$', path.read_text(encoding='utf-8'), flags=re.MULTILINE))
        module = importlib.import_module(package)
        classes = [getattr(module, name) for name in module.__all__]
        found += [
            f'{path.relative_to(root)}:1: no section for {item.__name__}'
            for item in classes
            if isinstance(item, type) and not inspect.isabstract(item) and item.__name__ not in headings
        ]
        for start, block in _python_blocks(path):
            found += [
                f'{path.relative_to(root)}:{start + line}: imports {module_name}'
                for line, module_name, _ in _imported_names(block)
                if module_name == SYSTEMS or module_name.startswith(f'{SYSTEMS}.')
            ]
    general = root / CORE_COMPONENTS / 'general.md'
    headings = set(re.findall(r'^## (\w+)$', general.read_text(encoding='utf-8'), flags=re.MULTILINE))
    for concept in pkgutil.iter_modules(importlib.import_module(GENERAL).__path__):
        module = importlib.import_module(f'{GENERAL}.{concept.name}')
        found += [
            f'{general.relative_to(root)}:1: no section for {name}' for name in module.__all__ if name not in headings
        ]
    return found


def check_troubleshooting_covers_every_error(root: Path) -> list[str]:
    """Return every public error class of the core without a section in the troubleshooting page."""
    page = root / USER_GUIDE / 'troubleshooting.md'
    headings = set(re.findall(r'^## (\w+)$', page.read_text(encoding='utf-8'), flags=re.MULTILINE))
    module = importlib.import_module('skcomplex.core.errors')
    errors = [
        name
        for name, item in vars(module).items()
        if isinstance(item, type)
        and issubclass(item, Exception)
        and not issubclass(item, Warning)
        and item.__module__ == module.__name__
        and not name.startswith('_')
    ]
    if not errors:
        return [f'{page.relative_to(root)}:1: the core defines no error class']
    return [f'{page.relative_to(root)}:1: no section for {name}' for name in errors if name not in headings]


def _is_private(name: str) -> bool:
    """Return whether a name is private, leaving out dunder names and the throwaway name `_`."""
    return name.startswith('_') and name != '_' and DUNDER.fullmatch(name) is None


def _private_uses(code: str) -> list[tuple[int, str]]:
    """Return every call or attribute read of a private name in a block of code, as its line and its expression.

    A name that the block defines itself, such as a hook of a subclass or its own helper, may be read through `self`,
    `cls`, or `super()`, and a function it defines may be called.
    """
    tree = ast.parse(code)
    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)}
    nodes = list(ast.walk(tree))
    attributes = [
        (node.lineno, ast.unparse(node))
        for node in nodes
        if isinstance(node, ast.Attribute)
        and _is_private(node.attr)
        and not (node.attr in defined and ast.unparse(node.value) in OWN_RECEIVERS)
    ]
    calls = [
        (node.lineno, node.func.id)
        for node in nodes
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and _is_private(node.func.id)
        and node.func.id not in defined
    ]
    return sorted(attributes + calls)


def _docstring_examples(path: Path) -> list[tuple[int, str]]:
    """Return the doctest examples of the docstrings of a module, each with the line of the file before it."""
    examples = []
    for node in ast.walk(_read_tree(path)):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        docstring = ast.get_docstring(node, clean=False)
        if docstring and '>>>' in docstring:
            start = node.body[0].lineno - 1
            examples += [(start + example.lineno, example.source) for example in DOCTESTS.get_examples(docstring)]
    return examples


def check_documents_show_only_public_names(root: Path) -> list[str]:
    """Return every private name that a docstring example, a document, or a gallery example calls or reads."""
    samples = [(path, *sample) for path in sorted((root / 'src').rglob('*.py')) for sample in _docstring_examples(path)]
    samples += [(root / document, *block) for document in _documents(root) for block in _python_blocks(root / document)]
    samples += [(path, 0, path.read_text(encoding='utf-8')) for path in sorted((root / GALLERY).rglob('*.py'))]
    return [
        f'{path.relative_to(root)}:{start + line}: uses the private name {name}'
        for path, start, code in samples
        for line, name in _private_uses(code)
    ]


def _imported_modules(path: Path, package: str) -> list[tuple[int, str]]:
    """Return the absolute names of the modules that a source file of a package imports, each with its line."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package.split('.')[: len(package.split('.')) - node.level + 1] if node.level else []
            module = '.'.join([*base, *([node.module] if node.module else [])])
            found += [(node.lineno, module)] + [(node.lineno, f'{module}.{alias.name}') for alias in node.names]
    return found


def check_core_never_imports_the_systems(root: Path) -> list[str]:
    """Return every import of `skcomplex.systems` by a module under `src/skcomplex/core`."""
    source = root / 'src'
    return [
        f'{path.relative_to(root)}:{line}: imports {module}'
        for path in sorted((source / 'skcomplex' / 'core').rglob('*.py'))
        for line, module in _imported_modules(path, '.'.join(path.relative_to(source).parent.parts))
        if module == SYSTEMS or module.startswith(f'{SYSTEMS}.')
    ]


def check_shared_packages_hold_only_subpackages(root: Path) -> list[str]:
    """Return every loose module at the root of `systems/general` or of a field `common`."""
    systems = root / SOURCE / 'systems'
    roots = [systems / 'general', *sorted(path.parent for path in systems.glob('*/common/__init__.py'))]
    return [
        f'{path.relative_to(root)}:1: loose module at the root of a shared package'
        for package in roots
        for path in sorted(package.glob('*.py'))
        if path.name != '__init__.py'
    ]


def _public_classes() -> list[type]:
    """Return the public classes of the subpackages of `skcomplex.core.components`, without typed dictionaries."""
    found: dict[str, type] = {}
    for name in ('elements', 'interactions', 'laws', 'spaces', 'specs', 'systems'):
        package = importlib.import_module(f'skcomplex.core.components.{name}')
        for item in package.__all__:
            value = getattr(package, item)
            if isinstance(value, type) and not is_typeddict(value):
                found[item] = value
    return list(found.values())


def _named_classes(path: Path) -> tuple[set[str], set[str]]:
    """Return the names a core module uses in its code, and the names of the bases of the classes it defines."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    bases = {
        base.id
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        for base in node.bases
        if isinstance(base, ast.Name)
    }
    return names, bases


def _is_structural_base(klass: type, classes: list[type]) -> bool:
    """Return whether a class is a base, by its name or because another public class of the core subclasses it."""
    return klass.__name__.startswith('Base') or any(
        other is not klass and issubclass(other, klass) for other in classes
    )


def _is_handled_by_type(klass: type, modules: dict[Path, tuple[set[str], set[str]]]) -> bool:
    """Return whether a core module other than its own and those of its subclasses names a class in its code."""
    own = Path(str(importlib.import_module(klass.__module__).__file__))
    return any(
        klass.__name__ in names and klass.__name__ not in bases
        for path, (names, bases) in modules.items()
        if path.resolve() != own.resolve()
    )


def _law_classes(klass: type) -> tuple[type, ...]:
    """Return the law classes an interaction class lists."""
    return cast(tuple[type, ...], cast(Any, klass)._get_law_classes())


def _is_general_form(klass: type, classes: list[type], interaction: type) -> bool:
    """Return whether an interaction lists the fewest law classes among those of its base that list one of its laws.

    `RuleLaw` counts for no class, since a law of arbitrary code is never a mathematical operation the engine lowers.
    """
    if not issubclass(klass, interaction):
        return False
    rule = importlib.import_module('skcomplex.core.components.laws').RuleLaw
    laws = _law_classes(klass)
    siblings = [
        _law_classes(other) for other in classes if issubclass(other, interaction) and other.__base__ is klass.__base__
    ]
    return any(len(laws) == min(len(listed) for listed in siblings if law in listed) for law in laws if law is not rule)


def check_core_components_obey_the_placement_rule(root: Path) -> list[str]:
    """Return every public class of the core components that is no base, is not handled by type, and is no form."""
    core = root / SOURCE / 'core'
    modules = {path: _named_classes(path) for path in sorted(core.rglob('*.py'))}
    classes = _public_classes()
    interaction = importlib.import_module('skcomplex.core.components.interactions').BaseInteraction
    return [
        f'{_locate(klass, root)}: {klass.__name__} is a recipe, not a core class'
        for klass in sorted(classes, key=lambda klass: klass.__name__)
        if not (
            _is_structural_base(klass, classes)
            or _is_handled_by_type(klass, modules)
            or _is_general_form(klass, classes, interaction)
        )
    ]


def _read_tree(path: Path) -> ast.Module:
    """Return the syntax tree of a Python file."""
    return ast.parse(path.read_text(encoding='utf-8'))


def _assigned_name(node: ast.stmt) -> str | None:
    """Return the one name a module assignment binds, or nothing for another statement."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _constant_name(node: ast.stmt) -> str | None:
    """Return the name a module constant binds, or nothing for another statement."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
        names = [item.id if isinstance(item, ast.Name) else '' for item in node.targets[0].elts]
        return ', '.join(names) if all(CONSTANT_NAME.fullmatch(name) for name in names) else None
    name = _assigned_name(node)
    return name if name is not None and CONSTANT_NAME.fullmatch(name) else None


def _read_names(node: ast.AST) -> set[str]:
    """Return the plain names an expression or a statement reads."""
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _is_docstring(node: ast.stmt) -> bool:
    """Return whether a statement is a bare string, as a docstring is."""
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _derived_indices(body: list[ast.stmt]) -> dict[int, int]:
    """Return the module assignments whose value reads a definition of the module, with the last statement they read.

    A derived value reads a class or a function of its module, or another derived value, so it cannot sit above it.
    """
    defined: dict[str, int] = {}
    derived: dict[int, int] = {}
    for index, node in enumerate(body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined[node.name] = index
            continue
        name = _assigned_name(node)
        value = getattr(node, 'value', None)
        read = [defined[read] for read in _read_names(value) if read in defined] if value is not None else []
        if name is not None and read:
            derived[index] = max(read)
            defined[name] = index
    return derived


def _misplaced_constants(path: Path) -> list[str]:
    """Return every module constant of a file that is private or does not lead its module, as `file:line: name`."""
    body = _read_tree(path).body
    last_import = max((index for index, node in enumerate(body) if isinstance(node, IMPORTS)), default=-1)
    derived = _derived_indices(body)
    found: list[str] = []
    for index, node in enumerate(body):
        name = _constant_name(node)
        if name is None:
            continue
        if name.startswith('_'):
            found.append(f'{path}:{node.lineno}: {name} is private')
        if index in derived:
            above = [other for other in range(derived[index] + 1, index) if other not in derived]
        else:
            above = [
                other
                for other in range(index)
                if not (
                    (other == 0 and _is_docstring(body[other]))
                    or other <= last_import
                    or (_constant_name(body[other]) is not None and other not in derived)
                )
            ]
        if above:
            found.append(f'{path}:{node.lineno}: {name} sits below line {body[above[0]].lineno}')
    return found


def _blank_and_comment_lines(path: Path) -> tuple[set[int], set[int]]:
    """Return the blank lines and the lines holding a comment of a Python file, read from its tokens."""
    blank: set[int] = set()
    comments: set[int] = set()
    with tokenize.open(path) as stream:
        previous = 0
        for token in tokenize.generate_tokens(stream.readline):
            if token.type == tokenize.COMMENT:
                comments.add(token.start[0])
            elif token.type == tokenize.NL and token.start[0] != previous:
                blank.add(token.start[0])
            if token.type not in (tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
                previous = token.end[0]
    return blank, comments


def _constant_gaps(path: Path) -> list[str]:
    """Return every gap of a block of module constants, as `file:line: what`."""
    body = _read_tree(path).body
    blank, comments = _blank_and_comment_lines(path)
    found: list[str] = []
    for index, node in enumerate(body):
        name = _constant_name(node)
        if name is None:
            continue
        follows = index > 0 and _constant_name(body[index - 1]) is not None
        start = (body[index - 1].end_lineno or node.lineno) + 1 if follows else node.lineno
        lines = range(start, (node.end_lineno or node.lineno) + 1)
        found += [f'{path}:{line}: blank line in the block of {name}' for line in lines if line in blank]
        found += [f'{path}:{line}: comment in the block of {name}' for line in lines if line in comments]
        if not follows and node.lineno - 1 in comments:
            found.append(f'{path}:{node.lineno - 1}: comment above the block of {name}')
        if index + 1 < len(body) and _is_docstring(body[index + 1]):
            found.append(f'{path}:{body[index + 1].lineno}: attribute docstring of {name}')
    return found


def _module_name(path: Path, source: Path) -> str:
    """Return the dotted name of a module of the sources."""
    parts = path.relative_to(source).with_suffix('').parts
    return '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)


def _imported_bindings(tree: ast.Module, module: str, is_package: bool) -> dict[str, str]:
    """Return the names a module binds by its imports, each with the dotted name it binds."""
    package = module if is_package else module.rpartition('.')[0]
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.partition('.')[0]] = (
                    alias.name if alias.asname else alias.name.partition('.')[0]
                )
        elif isinstance(node, ast.ImportFrom):
            base = package.split('.')[: len(package.split('.')) - node.level + 1] if node.level else []
            source = '.'.join([*base, *([node.module] if node.module else [])])
            for alias in node.names:
                bindings[alias.asname or alias.name] = f'{source}.{alias.name}'
    return bindings


def _is_compiled(node: ast.FunctionDef, bindings: dict[str, str]) -> bool:
    """Return whether a function carries a decorator of Numba, in any import form."""
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        while isinstance(target, ast.Attribute):
            target = target.value
        if isinstance(target, ast.Name) and bindings.get(target.id, '').split('.')[0] == 'numba':
            return True
    return False


def _read_targets(node: ast.AST, bindings: dict[str, str], module: str, local: set[str]) -> set[str]:
    """Return the dotted names of the module functions a statement reads, of its own module or imported."""
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            if child.id in local:
                found.add(f'{module}.{child.id}')
            elif child.id in bindings:
                found.add(bindings[child.id])
        elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name) and child.value.id in bindings:
            found.add(f'{bindings[child.value.id]}.{child.attr}')
    return found


def _function_readers(source: Path) -> tuple[dict[str, set[str]], dict[str, ast.FunctionDef], set[str], set[str]]:
    """Return the readers of every module function of the sources, its definition, the compiled ones, and `__all__`.

    A reader is named `module.Name` for a class or a function, and `module:line` for another module statement.
    """
    readers: dict[str, set[str]] = {}
    functions: dict[str, ast.FunctionDef] = {}
    compiled: set[str] = set()
    exported: set[str] = set()
    for path in sorted(source.rglob('*.py')):
        module = _module_name(path, source)
        tree = _read_tree(path)
        bindings = _imported_bindings(tree, module, path.name == '__init__.py')
        local = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                functions[f'{module}.{node.name}'] = node
                if _is_compiled(node, bindings):
                    compiled.add(f'{module}.{node.name}')
            if _assigned_name(node) == '__all__':
                exported |= {
                    item.value
                    for item in ast.walk(node)
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                }
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for part, reader in _reading_parts(node, module):
                for target in _read_targets(part, bindings, module, local):
                    if target != reader:
                        readers.setdefault(target, set()).add(reader)
    return readers, functions, compiled, exported


def _reading_parts(node: ast.stmt, module: str) -> list[tuple[ast.AST, str]]:
    """Return the parts of a module statement that read names, each with the reader it stands for.

    The methods of a class read as the class. Its other statements run when the class is built, so they read as a value,
    `module:line`, as a module statement does.
    """
    if isinstance(node, ast.FunctionDef):
        return [(node, f'{module}.{node.name}')]
    if not isinstance(node, ast.ClassDef):
        return [(node, f'{module}:{node.lineno}')]
    parts: list[tuple[ast.AST, str]] = []
    for child in node.body:
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            parts.append((child, f'{module}.{node.name}'))
        else:
            parts.append((child, f'{module}:{child.lineno}'))
    return parts


def _class_readers(name: str, readers: dict[str, set[str]], functions: dict[str, ast.FunctionDef]) -> set[str]:
    """Return the classes that read a module function, directly or through other private module functions."""
    found: set[str] = set()
    seen = {name}
    pending = [name]
    while pending:
        for reader in readers.get(pending.pop(), set()):
            if reader in seen or ':' in reader:
                continue
            seen.add(reader)
            if reader not in functions:
                found.add(reader)
            elif reader.rpartition('.')[2].startswith('_'):
                pending.append(reader)
    return found


def _is_value_read(name: str, readers: dict[str, set[str]], functions: dict[str, ast.FunctionDef]) -> bool:
    """Return whether a module value reads a module function, directly or through other private module functions."""
    seen = {name}
    pending = [name]
    while pending:
        for reader in readers.get(pending.pop(), set()):
            if ':' in reader:
                return True
            if reader in functions and reader not in seen and reader.rpartition('.')[2].startswith('_'):
                seen.add(reader)
                pending.append(reader)
    return False


def _unowned_functions(source: Path, packages: tuple[str, ...]) -> list[str]:
    """Return every module function of the packages that a class should own, or that no package re-exports."""
    readers, functions, compiled, exported = _function_readers(source)
    found: list[str] = []
    for name, node in functions.items():
        module, _, function = name.rpartition('.')
        if not module.startswith(packages) or name in compiled:
            continue
        if function.startswith('_'):
            classes = _class_readers(name, readers, functions)
            derived = _is_value_read(name, readers, functions)
            if classes and not derived:
                found.append(f'{module}:{node.lineno}: {function} is read by {", ".join(sorted(classes))}')
        elif '._' in f'.{module}' and function not in exported and not UTILS_MODULE.search(module):
            found.append(f'{module}:{node.lineno}: {function} is public in a private module and not re-exported')
    return found


def _is_property(node: ast.FunctionDef) -> bool:
    """Return whether a method is a property, or its setter or deleter."""
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id in PROPERTIES:
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr in ('setter', 'getter', 'deleter', *PROPERTIES):
            return True
    return False


def _foreign_names(module: str, qualname: str) -> set[str]:
    """Return the names that the bases of a class from another library define."""
    value: object = importlib.import_module(module)
    for part in qualname.split('.'):
        value = getattr(value, part, None)
    if not isinstance(value, type):
        return set()
    return {name for base in value.__mro__[1:] if not base.__module__.startswith(PACKAGE) for name in vars(base)}


def _methods(tree: ast.Module) -> list[tuple[str, ast.FunctionDef]]:
    """Return every method of a module, each with the qualified name of its class."""
    found: list[tuple[str, ast.FunctionDef]] = []
    pending: list[tuple[str, ast.stmt]] = [('', node) for node in tree.body]
    while pending:
        prefix, node = pending.pop(0)
        if isinstance(node, ast.ClassDef):
            qualname = f'{prefix}{node.name}'
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    found.append((qualname, child))
                elif isinstance(child, ast.ClassDef):
                    pending.append((f'{qualname}.', child))
    return found


def _is_exempt_method(name: str) -> bool:
    """Return whether a method name is a magic method or a hook a library calls by a fixed name, as `_eval_`."""
    return bool(DUNDER.fullmatch(name) or LIBRARY_HOOK.fullmatch(name))


def _nonverb_methods(path: Path, module: str) -> list[str]:
    """Return every method of a module whose name does not begin with a verb, as `file:line: Class.method`."""
    found: list[str] = []
    foreign: dict[str, set[str]] = {}
    for qualname, node in _methods(_read_tree(path)):
        if _is_exempt_method(node.name) or _is_property(node) or node.name.strip('_').split('_')[0] in VERBS:
            continue
        if qualname not in foreign:
            foreign[qualname] = _foreign_names(module, qualname)
        if node.name not in foreign[qualname]:
            found.append(f'{path}:{node.lineno}: {qualname}.{node.name}')
    return found


def _sage_function_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Return the names that bind Sage's function factory, and the names that bind a Sage module."""
    factories: set[str] = set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('sage'):
            factories |= {alias.asname or alias.name for alias in node.names if alias.name == 'function'}
        elif isinstance(node, ast.Import):
            modules |= {alias.asname or alias.name for alias in node.names if alias.name.startswith('sage')}
    return factories, modules


def _factory_calls(path: Path) -> list[str]:
    """Return every call of Sage's function factory that defines a library function, as `file:line`."""
    tree = _read_tree(path)
    factories, modules = _sage_function_names(tree)
    found: list[str] = []
    for statement in tree.body:
        top = not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            is_factory = (isinstance(target, ast.Name) and target.id in factories) or (
                isinstance(target, ast.Attribute)
                and target.attr == 'function'
                and isinstance(target.value, ast.Name)
                and target.value.id in modules
            )
            hooks = [keyword.arg for keyword in node.keywords if keyword.arg and FACTORY_KEYWORD.fullmatch(keyword.arg)]
            if is_factory and (top or hooks):
                found.append(f'{path}:{node.lineno}: function factory {"at module level" if top else ", ".join(hooks)}')
    return found


def _python_files(folders: tuple[str, ...], root: Path) -> list[Path]:
    """Return the Python files of the checked folders, without the generated pages."""
    return [
        path
        for folder in folders
        for path in sorted((root / folder).rglob('*.py'))
        if not path.relative_to(root).is_relative_to(GENERATED)
    ]


def _checked_modules(packages: tuple[str, ...], root: Path) -> list[tuple[Path, str]]:
    """Return the modules of the checked packages, each with its dotted name."""
    source = root / 'src'
    folders = tuple(str(Path('src', *package.split('.'))) for package in packages)
    return [(path, _module_name(path, source)) for path in _python_files(folders, root)]


def check_module_constants_lead_their_module(root: Path) -> list[str]:
    """Return every module constant that is private or sits outside the block after the imports or what it reads."""
    return [line for path in _python_files(FOLDERS, root) for line in _misplaced_constants(path.relative_to(root))]


def check_constant_blocks_hold_no_gap(root: Path) -> list[str]:
    """Return every blank line, comment, or attribute docstring in or right above a block of constants."""
    return [line for path in _python_files(FOLDERS, root) for line in _constant_gaps(path.relative_to(root))]


def check_classes_own_the_functions_they_read(root: Path) -> list[str]:
    """Return every module function a class reads that is no method, apart from kernels and helpers of functions."""
    found = []
    for line in _unowned_functions(root / 'src', (PACKAGE,)):
        module, _, rest = line.partition(':')
        path = Path('src', *module.split('.'))
        found.append(f'{path / "__init__.py" if (root / path).is_dir() else path.with_suffix(".py")}:{rest}')
    return found


def check_methods_begin_with_a_verb(root: Path) -> list[str]:
    """Return every method that does not begin with a verb, apart from magic methods, library hooks, and properties."""
    modules = _checked_modules((PACKAGE,), root)
    return [line for path, module in modules for line in _nonverb_methods(path.relative_to(root), module)]


def check_symbolic_functions_are_builtin_subclasses(root: Path) -> list[str]:
    """Return every call of Sage's function factory but one that declares an unknown function inside a function."""
    modules = _checked_modules((PACKAGE,), root)
    return [line for path, _ in modules for line in _factory_calls(path.relative_to(root))]


def _library_modules(packages: tuple[str, ...]) -> Iterator[ModuleType]:
    """Return every module of the packages, the packages themselves first."""
    for name in packages:
        package = importlib.import_module(name)
        yield package
        for info in pkgutil.walk_packages(package.__path__, f'{name}.'):
            yield importlib.import_module(info.name)


def _importable_modules(package: str, skipped: Callable[[str], bool]) -> Iterator[ModuleType]:
    """Return every module of a package that imports, leaving out the names the predicate skips."""
    loaded = importlib.import_module(package)
    names = [loaded.__name__] + [info.name for info in pkgutil.walk_packages(loaded.__path__, f'{package}.')]
    for name in names:
        if skipped(name):
            continue
        try:
            yield importlib.import_module(name)
        except ImportError:
            continue


def _hand_written(cls: type) -> list[str]:
    """Return the magic methods a class defines itself in source code, leaving out those Python generates.

    Dataclasses generate methods, and from Python 3.14 the annotations of a class compile to `__annotate_func__`.
    """
    source = str(Path(str(importlib.import_module(PACKAGE).__file__)).parent)
    names = []
    for name, value in vars(cls).items():
        if not (name.startswith('__') and name.endswith('__')) or not callable(value) or name == '__annotate_func__':
            continue
        code = getattr(value, '__code__', None)
        if code is not None and code.co_filename.startswith(source):
            names.append(name)
    return names


def check_core_names_carry_no_field_term(root: Path) -> list[str]:
    """Return every public name of a package of `skcomplex.core.components` that contains a term of a field."""
    found = []
    for module in _library_modules(CLASS_PACKAGES):
        if not module.__name__.startswith('skcomplex.core.components'):
            continue
        for name in getattr(module, '__all__', []):
            terms = sorted(set(re.findall('[A-Z][a-z]*', name)) & set(FIELD_TERMS))
            if terms:
                found.append(f'{_locate(getattr(module, name), root)}: {module.__name__}.{name} holds {terms}')
    return found


def check_no_package_exports_a_molecular_name(root: Path) -> list[str]:
    """Return every name of RNA, proteins, or hydrophobicity that the core, the engine, or the analysis exports."""
    prefixes = ('skcomplex.core.components', 'skcomplex.core.engine', 'skcomplex.core.analysis')
    return [
        f'{_locate(getattr(module, name), root)}: {module.__name__}.{name} is a molecular name'
        for module in _library_modules(CLASS_PACKAGES)
        if module.__name__.startswith(prefixes)
        for name in getattr(module, '__all__', [])
        if any(
            term in re.sub(r'[^a-z]', ' ', re.sub(r'([A-Z])', r' \1', name).lower()).split() for term in MOLECULAR_TERMS
        )
    ]


def check_no_class_defines_magic_methods(root: Path) -> list[str]:
    """Return every class of the library that defines a magic method but `__init__`."""
    found = []
    for module in _library_modules(CLASS_PACKAGES):
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            extra = set(_hand_written(cls)) - {'__init__'}
            if extra:
                found.append(f'{_locate(cls, root)}: {cls.__qualname__} defines {sorted(extra)}')
    return found


def check_no_module_calls_a_magic_method(root: Path) -> list[str]:
    """Return every read of a magic method such as `__new__` or `__getitem__` of an object by the library."""
    return [
        f'{path.relative_to(root)}:{number}: reads {name}'
        for path in sorted((root / SOURCE).rglob('*.py'))
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1)
        for name in re.findall(r'\.(__[a-z_]+__)\b', line)
        if name not in ALLOWED_MAGIC_READS
    ]


def _core_modules(root: Path) -> Iterator[tuple[Path, str, ast.Module]]:
    """Return the path, the dotted name, and the syntax tree of every module of the core and the shared recipes."""
    source = root / SOURCE
    for path in sorted([*(source / 'core').rglob('*.py'), *(source / 'systems' / 'general').rglob('*.py')]):
        yield path.relative_to(root), _module_name(path, source.parent), _read_tree(path)


def _docstrings(root: Path) -> Iterator[tuple[str, str, str, bool]]:
    """Return the place, the dotted name, the docstring, and the publicity of every docstring of `skcomplex.core`.

    The walk descends into classes, not into functions.
    """
    for path, module, tree in _core_modules(root):
        yield from _walk_docstrings(tree, path, module, public=True)


def _walk_docstrings(node: ast.AST, path: Path, name: str, *, public: bool) -> Iterator[tuple[str, str, str, bool]]:
    """Return the place, the dotted name, the docstring, and the publicity of a node and of the definitions in it."""
    docstring = ast.get_docstring(node) if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef) else None
    if docstring:
        line = node.lineno if isinstance(node, ast.ClassDef | ast.FunctionDef) else 1
        yield f'{path}:{line}', name, docstring, public
    if isinstance(node, ast.Module | ast.ClassDef):
        for child in node.body:
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                is_public = public and not child.name.startswith('_')
                yield from _walk_docstrings(child, path, f'{name}.{child.name}', public=is_public)


def _public_definitions(root: Path, kind: type[ast.stmt]) -> Iterator[tuple[str, str, str, str]]:
    """Return the place, the dotted name, the name, and the docstring of every top-level public class or function."""
    for path, module, tree in _core_modules(root):
        for node in tree.body:
            if not isinstance(node, kind) or not isinstance(node, ast.ClassDef | ast.FunctionDef):
                continue
            docstring = ast.get_docstring(node)
            if docstring and not node.name.startswith('_'):
                yield f'{path}:{node.lineno}', f'{module}.{node.name}', node.name, docstring


def _prose_lines(docstring: str) -> list[str]:
    """Return the prose lines of a docstring, stripped, with doctest code and inline code removed.

    A doctest block runs from a line that starts with `>>>` to the next blank line, so its outputs are exempt too.
    """
    lines = []
    in_code = False
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped.startswith(('>>>', '...')):
            in_code = True
        elif not stripped:
            in_code = False
        lines.append('' if in_code else stripped)
    text = re.sub('`[^`]*`', lambda match: '\n' * match.group().count('\n'), '\n'.join(lines))
    return text.splitlines()


def check_docstring_summaries_are_short(root: Path) -> list[str]:
    """Return every docstring of `skcomplex.core` whose summary line is longer than 80 characters."""
    return [
        f'{place}: the summary of {name} is {len(docstring.splitlines()[0])} characters long'
        for place, name, docstring, _ in _docstrings(root)
        if len(docstring.splitlines()[0]) > SUMMARY_LENGTH
    ]


def check_docstrings_read_naturally(root: Path) -> list[str]:
    """Return every docstring of the core with a semicolon, and every public one with a dash or a banned phrase."""
    found = []
    for place, name, docstring, public in _docstrings(root):
        for line in _prose_lines(docstring):
            faults = _prose_faults(line) if public else ['semicolon'] * (';' in line)
            if faults:
                found.append(f'{place}: {name}: {", ".join(faults)}: {line}')
    return found


def check_class_summaries_name_their_kind(root: Path) -> list[str]:
    """Return every public class that names no kind after an article, and every summary that holds a field term."""
    found = []
    for place, name, short, docstring in _public_definitions(root, ast.ClassDef):
        summary = docstring.splitlines()[0]
        kind = next((word for suffix, word in KINDS.items() if short.endswith(suffix)), None)
        if not summary.startswith(('A ', 'An ')) or (kind is not None and kind not in summary.lower()):
            found.append(f'{place}: {name}: {summary}')
    definitions = [*_public_definitions(root, ast.ClassDef), *_public_definitions(root, ast.FunctionDef)]
    for place, name, _, docstring in definitions:
        words = {word.lower() for word in re.findall('[A-Za-z]+', docstring.splitlines()[0])}
        terms = sorted(term for term in FIELD_TERMS if term.lower() in words)
        if terms:
            found.append(f'{place}: {name} holds {terms}')
    return found


def check_docstrings_say_main_public_methods(root: Path) -> list[str]:
    """Return every docstring that calls the main public methods verbs."""
    return [
        f'{place}: {name} says verbs'
        for place, name, docstring, _ in _docstrings(root)
        if re.search(r'\bverbs?\b', docstring, flags=re.IGNORECASE)
    ]


def check_components_compare_and_name_uses(root: Path) -> list[str]:
    """Return every public class of a family that compares itself with no nearest part or names no uses."""
    found = []
    for place, name, _, docstring in _public_definitions(root, ast.ClassDef):
        if not name.startswith(FAMILIES):
            continue
        paragraphs = [' '.join(paragraph.split()) for paragraph in docstring.split('\n\n')]
        compares = any(p.startswith('Unlike') and 'Choose it when' in p for p in paragraphs)
        names_uses = any(p.startswith('It appears in') for p in paragraphs)
        if not (compares and names_uses):
            found.append(f'{place}: {name} lacks its comparison or its uses')
    return found


def check_every_error_is_raised_in_the_library(root: Path) -> list[str]:
    """Return every error class that the library never builds, nor a subclass of it, where it reports a failure."""
    errors = importlib.import_module('skcomplex.core.errors')
    paths = (root / SOURCE).rglob('*.py')
    source = '\n'.join(path.read_text(encoding='utf-8') for path in paths if path.name != 'errors.py')
    base = errors.SkcomplexError
    return [
        f'{_locate(item, root)}: {name} is never raised'
        for name, item in vars(errors).items()
        if isinstance(item, type)
        and issubclass(item, base)
        and not item.__subclasses__()
        and not re.search(rf'(?<!class )\b{name}\(|\berror={name}\b', source)
    ]


def check_every_dataclass_is_a_frozen_spec_or_result(root: Path) -> list[str]:
    """Return every dataclass of the core that is not frozen, or no spec and no result a function returns."""
    spec = importlib.import_module('skcomplex.core.specs').BaseSpec
    found = []
    for module in _importable_modules('skcomplex.core', lambda _: False):
        for value in vars(module).values():
            own = isinstance(value, type) and value.__module__ == module.__name__
            if not own or not isinstance(value, type) or not dataclasses.is_dataclass(value):
                continue
            kind = issubclass(value, spec) or (value.__name__.endswith('Result') and value.__bases__ == (object,))
            if not (kind and cast(Any, value).__dataclass_params__.frozen):
                found.append(f'{_locate(value, root)}: {module.__name__}.{value.__qualname__}')
    return found


def check_enums_and_protocols_live_in_their_modules(root: Path) -> list[str]:
    """Return every enum outside an `enums` module and every protocol outside a `types` module, the systems aside."""

    def skipped(name: str) -> bool:
        """Return whether a module is a system module."""
        return name.startswith('skcomplex.systems')

    def misplaced(value: type, module: str) -> bool:
        """Return whether an enum or a protocol is defined outside the module of its kind."""
        if issubclass(value, enum.Enum):
            return not module.endswith('.enums')
        return getattr(value, '_is_protocol', False) and not module.endswith('.types')

    return [
        f'{_locate(value, root)}: {module.__name__}.{value.__qualname__}'
        for module in _importable_modules(PACKAGE, skipped)
        for value in vars(module).values()
        if inspect.isclass(value) and value.__module__ == module.__name__ and misplaced(value, module.__name__)
    ]


def check_every_base_derives_from_the_root(root: Path) -> list[str]:
    """Return every class named `Base*` outside the systems that derives from neither `BaseConfig` nor `BaseSpec`."""
    config = importlib.import_module('skcomplex.core.base').BaseConfig
    spec = importlib.import_module('skcomplex.core.specs').BaseSpec
    return [
        f'{_locate(value, root)}: {module.__name__}.{name}'
        for module in _importable_modules(PACKAGE, lambda name: name.startswith('skcomplex.systems'))
        for name, value in vars(module).items()
        if inspect.isclass(value)
        and value.__module__ == module.__name__
        and name.lstrip('_').startswith('Base')
        and not issubclass(value, config | spec | BaseException)
    ]


def _generated_classes(root: Path) -> dict[str, list[int]]:
    """Return the `__signature__` assignments and the calls of `type` with three arguments of a source tree."""
    found: dict[str, list[int]] = {}
    for path in sorted(root.rglob('*.py')):
        lines = []
        for node in ast.walk(ast.parse(path.read_text())):
            is_call = isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'type'
            is_type = is_call and len(cast(ast.Call, node).args) == TYPE_ARGUMENTS
            is_signature = '__signature__' in {getattr(node, 'attr', None), getattr(node, 'value', None)}
            if is_type or is_signature:
                lines.append(cast(ast.expr, node).lineno)
        if lines:
            found[path.relative_to(root).as_posix()] = sorted(lines)
    return found


def check_library_generates_no_classes_of_components(root: Path) -> list[str]:
    """Return every `__signature__` assignment and every class built with `type` but the agents of Mesa."""
    found = _generated_classes(root / SOURCE)
    return [
        f'{SOURCE / path}:{lines[0]}: {len(lines)} generated classes or signatures'
        for path, lines in found.items()
        if len(lines) != GENERATED_CLASSES_ALLOWED.get(path)
    ] + [
        f'{SOURCE / path}:1: expected {count} generated classes of Mesa agents'
        for path, count in GENERATED_CLASSES_ALLOWED.items()
        if path not in found
    ]


def _part_words(text: str, first: int) -> Iterator[tuple[int, str]]:
    """Return the line and the text of every use of part in a text that no kept phrase covers.

    The text starts at the line `first`. A kept phrase may run over the end of a line, since prose wraps.
    """
    flat = text.replace('\n', ' ')
    for match in PART_WORD.finditer(flat):
        start, end = match.span()
        window = (max(0, start - PART_CONTEXT), end + PART_CONTEXT)
        kept = any(
            found.start() <= start and found.end() >= end
            for phrase in KEPT_PART_PHRASES
            for found in phrase.finditer(flat, *window)
        )
        if not kept:
            line = first + text.count('\n', 0, start)
            yield line, text.splitlines()[line - first].strip()


def _part_names(path: Path) -> Iterator[tuple[int, str]]:
    """Return the line and the name of every class, function, argument, or assigned name with the word part."""
    for node in ast.walk(_read_tree(path)):
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            name = node.name
        elif isinstance(node, ast.arg):
            name = node.arg
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            name = node.id
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            name = node.attr
        else:
            continue
        words = {word.lower() for word in NAME_WORD.findall(name)}
        if words & {'part', 'parts'} and name not in KEPT_PART_NAMES:
            yield node.lineno, name


def _code_free(line: str) -> str:
    """Return a line of prose with its inline code replaced by the word `code`."""
    return INLINE_CODE.sub('code', line)


def _docstring_prose(path: Path) -> Iterator[tuple[int, str]]:
    """Return every docstring of a module with the line of its first line, its doctests and inline code removed."""
    for node in ast.walk(_read_tree(path)):
        body = getattr(node, 'body', None)
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) or not body:
            continue
        if not _is_docstring(body[0]):
            continue
        constant = cast(ast.Constant, cast(ast.Expr, body[0]).value)
        lines = []
        in_code = False
        for line in str(constant.value).splitlines():
            stripped = line.strip()
            in_code = stripped.startswith(('>>>', '...')) or (in_code and bool(stripped))
            lines.append('' if in_code else _code_free(line))
        yield constant.lineno, '\n'.join(lines)


def _comment_prose(path: Path) -> Iterator[tuple[int, str]]:
    """Return every comment of a module with its line and inline code removed."""
    with path.open('rb') as file:
        for token in tokenize.tokenize(file.readline):
            if token.type == tokenize.COMMENT:
                yield token.start[0], _code_free(token.string)


def _page_prose(path: Path) -> str:
    """Return the prose of a Markdown page, line for line, with fenced blocks blanked and inline code replaced."""
    return '\n'.join(_code_free(line) for _, line in _markdown_prose_lines(path.read_text(encoding='utf-8')))


def _markdown_prose_lines(text: str) -> list[tuple[int, str]]:
    """Return every line of a Markdown text with its number, with the lines of fenced blocks blank."""
    lines = []
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            in_fence = not in_fence
            lines.append((number, ''))
        else:
            lines.append((number, '' if in_fence else line))
    return lines


def _parts_of(root: Path, sources: list[Path], tests: list[Path], pages: list[Path]) -> list[str]:
    """Return every use of part for a component in names, docstrings, comments, and pages, as `file:line: text`."""
    found = []
    for path in [*sources, *tests]:
        named = _part_names(path)
        found += [f'{path.relative_to(root)}:{line}: part names a component: {item}' for line, item in named]
    for path in sources:
        prose = [*_docstring_prose(path), *_comment_prose(path)]
        found += [
            f'{path.relative_to(root)}:{line}: part names a component: {text}'
            for first, block in prose
            for line, text in _part_words(block, first)
        ]
    for path in pages:
        found += [
            f'{path.relative_to(root)}:{line}: part names a component: {text}'
            for line, text in _part_words(_page_prose(path), 1)
        ]
    return found


def check_components_are_not_called_parts(root: Path) -> list[str]:
    """Return every name, docstring, comment, or page line that calls a building block of a system a part."""
    sources = sorted((root / SOURCE).rglob('*.py'))
    tests = sorted((root / TESTS).rglob('*.py'))
    pages = [root / README, root / 'CONTRIBUTING.md', *sorted((root / 'docs' / 'overview').rglob('*.md'))]
    return _parts_of(root, sources, tests, pages)


def _relative_imports(path: Path) -> set[Path]:
    """Return the modules that a module imports relatively, through a package's re-exports too."""
    found = set()
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue
        target = path.parents[node.level - 1].joinpath(*(node.module or '').split('.'))
        if target.with_suffix('.py').is_file():
            found.add(target.with_suffix('.py'))
        elif (target / '__init__.py').is_file():
            names = {alias.name for alias in node.names}
            for item in ast.parse((target / '__init__.py').read_text()).body:
                if isinstance(item, ast.ImportFrom) and item.level == 1 and names & {a.name for a in item.names}:
                    found.add(target.joinpath(*str(item.module).split('.')).with_suffix('.py'))
    return found


def _field_imports(path: Path, field: Path) -> set[Path]:
    """Return the modules of a field that a module imports relatively, through a package's re-exports too."""
    return {module for module in _relative_imports(path) if module.is_relative_to(field) and module != path}


def _defines_field(node: ast.ClassDef) -> bool:
    """Return whether a class body gives the class variable `field`, which makes the class a system class."""
    return any(
        isinstance(item, ast.Assign) and any(getattr(target, 'id', None) == 'field' for target in item.targets)
        for item in node.body
    )


def _defines_system(path: Path) -> bool:
    """Return whether a module defines a system class."""
    return any(isinstance(node, ast.ClassDef) and _defines_field(node) for node in ast.parse(path.read_text()).body)


def _owners(field: Path) -> dict[Path, set[Path]]:
    """Return the system modules that reach every other private module of a field, through modules defining none.

    Public modules, such as the `types` module of a subfield, are part of the interface and serve no single system. A
    module of `common` that another module of `common` imports belongs to a shared family, such as the force tables of
    an external engine, and serves no single system either.
    """
    modules = [path for path in sorted(field.rglob('_*.py')) if path.name != '__init__.py']
    imports = {path: _field_imports(path, field) for path in modules}
    systems = {path for path in modules if _defines_system(path)}
    common = field / 'common'
    family = {
        module
        for path in modules
        if path.is_relative_to(common)
        for module in imports[path]
        if module.is_relative_to(common)
    }
    owners: dict[Path, set[Path]] = {path: set() for path in modules if path not in systems | family}
    for system in systems:
        stack, seen = list(imports[system]), set()
        while stack:
            module = stack.pop()
            if module in seen or module in systems or module not in imports:
                continue
            seen.add(module)
            if module in owners:
                owners[module].add(system)
            stack.extend(imports[module])
    return owners


def check_no_module_serves_one_system(root: Path) -> list[str]:
    """Return every module of a field that defines no system and serves only one system, which should hold it."""
    found = []
    for field in FIELDS:
        folder = root / SOURCE / 'systems' / field
        found += [
            f'{module.relative_to(root)}:1: serves only {next(iter(systems)).relative_to(root)}'
            for module, systems in _owners(folder).items()
            if len(systems) == 1
        ]
    return found


def check_no_system_module_imports_another(root: Path) -> list[str]:
    """Return every module defining systems that imports another module defining systems of its subfield."""
    found = []
    for field in FIELDS:
        folder = root / SOURCE / 'systems' / field
        systems = [path for path in sorted(folder.rglob('_*.py')) if _defines_system(path)]
        found += [
            f'{path.relative_to(root)}:1: imports {module.relative_to(root)}'
            for path in systems
            for module in sorted(_field_imports(path, folder))
            if module.parent == path.parent and _defines_system(module)
        ]
    return found


def _general_users(systems: Path) -> dict[Path, set[str]]:
    """Return the fields that reach every module of `systems/general`, directly or through other general modules."""
    general = systems / 'general'
    modules = [path for path in sorted(general.rglob('_*.py')) if path.name != '__init__.py']
    importers: dict[Path, set[Path]] = {path: set() for path in modules}
    for path in sorted(systems.rglob('*.py')):
        for module in _relative_imports(path):
            if module in importers and module != path:
                importers[module].add(path)
    users: dict[Path, set[str]] = {}
    for module in modules:
        stack, seen, fields = [module], set(), set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            for importer in importers.get(current, set()):
                if importer.is_relative_to(general):
                    stack.append(importer)
                else:
                    fields.add(importer.relative_to(systems).parts[0])
        users[module] = fields
    return users


def check_every_general_module_serves_two_fields(root: Path) -> list[str]:
    """Return every module of `systems/general` that has importers in fewer than two fields."""
    return [
        f'{module.relative_to(root)}:1: serves only {sorted(fields)}'
        for module, fields in _general_users(root / SOURCE / 'systems').items()
        if len(fields & set(FIELDS)) < SERVED_FIELDS
    ]


def check_connectors_live_in_the_library(root: Path) -> list[str]:
    """Return every module of the core components or the engine that names the electrical connector."""
    return [
        f'{path.relative_to(root)}:{number}: names the electrical connector'
        for package in CONNECTOR_PACKAGES
        for path in sorted((root / SOURCE / package).rglob('*.py'))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if "'electrical'" in line
    ]


def _mirrored_module(path: Path) -> Path | None:
    """Return the module a test file tests, or nothing when no module of the package carries its name."""
    name = path.stem.removeprefix('test_')
    folder = SOURCE / path.relative_to(TESTS).parent
    return next((module for module in (folder / f'_{name}.py', folder / f'{name}.py') if module.is_file()), None)


def check_tests_mirror_the_sources(root: Path) -> list[str]:
    """Return every test file that mirrors no module, as `tests/<package>/test_<module>.py` does `_<module>.py`."""
    return [
        f'{path.relative_to(root)}:1: mirrors no module of {SOURCE / path.relative_to(root / TESTS).parent}'
        for path in sorted((root / TESTS).rglob('test_*.py'))
        if _mirrored_module(path.relative_to(root)) is None
    ]


def _write_module(folder: Path, name: str, code: str) -> Path:
    """Write a module of code to a folder and return its path."""
    path = folder / f'{name}.py'
    path.write_text(textwrap.dedent(code), encoding='utf-8')
    return path


def _expected(name: str, found: object, expected: object) -> list[str]:
    """Return a finding of the script itself when a reader found something other than what it should."""
    return [] if found == expected else [f'{SCRIPT}:1: the reader {name} found {found!r}, not {expected!r}']


def check_misplaced_constants_are_named(folder: Path) -> list[str]:
    """Return a finding unless a private constant and a constant below a function are named, and a derived value not."""
    path = _write_module(
        folder,
        'constants',
        """
        import re
        LIMIT = 3
        def double(value):
            return 2 * value
        TWICE = double(LIMIT)
        _HIDDEN = 1
        """,
    )
    expected = [f'{path}:7: _HIDDEN is private', f'{path}:7: _HIDDEN sits below line 4']
    return _expected('_misplaced_constants', _misplaced_constants(path), expected)


def check_constant_gaps_are_named(folder: Path) -> list[str]:
    """Return a finding unless a blank line, a comment, and an attribute docstring of a block are named."""
    path = _write_module(
        folder,
        'gaps',
        """
        LIMIT = 3

        # The size.
        SIZE = 4
        \"\"\"The size.\"\"\"
        """,
    )
    tight = _write_module(folder, 'tight', 'LIMIT = 3\nSIZE = (\n    4,\n)\n')
    expected = [
        f'{path}:3: blank line in the block of SIZE',
        f'{path}:4: comment in the block of SIZE',
        f'{path}:6: attribute docstring of SIZE',
    ]
    return _expected('_constant_gaps', _constant_gaps(path), expected) + _expected(
        '_constant_gaps',
        _constant_gaps(tight),
        [],
    )


def check_unowned_functions_are_named(folder: Path) -> list[str]:
    """Return a finding unless a helper of a class and an unexported public function are named, and kernels are not."""
    package = folder / 'pack'
    package.mkdir()
    _write_module(package, '__init__', "__all__ = ['run']")
    _write_module(
        package,
        '_module',
        """
        from numba import njit
        @njit
        def _kernel(value):
            return value
        def _helper(value):
            return _kernel(value)
        def _of_run(value):
            return value
        def _of_table(value):
            return value
        TABLE = {'one': _of_table}
        def run(value):
            return _of_run(value)
        def spare(value):
            return value
        class Model:
            def fit(self, value):
                return _helper(value) + TABLE['one'](value)
        """,
    )
    expected = [
        'pack._module:6: _helper is read by pack._module.Model',
        'pack._module:15: spare is public in a private module and not re-exported',
    ]
    return _expected('_unowned_functions', _unowned_functions(folder, ('pack',)), expected)


def check_nonverb_methods_are_named(folder: Path) -> list[str]:
    """Return a finding unless a method that begins with a noun is named, and verbs, hooks, and properties are not."""
    path = _write_module(
        folder,
        'methods',
        """
        class Model:
            def __init__(self):
                self.size = 1
            def _eval_(self):
                return 0
            @property
            def total(self):
                return 0
            def get_total(self):
                return 0
            def _noise(self):
                return 0
        """,
    )
    sys.path.insert(0, str(folder))
    try:
        found = _nonverb_methods(path, 'methods')
    finally:
        sys.path.remove(str(folder))
        sys.modules.pop('methods', None)
    return _expected('_nonverb_methods', found, [f'{path}:12: Model._noise'])


def check_factory_calls_are_named(folder: Path) -> list[str]:
    """Return a finding unless a factory call at module level or with a hook is named, and an unknown is not."""
    path = _write_module(
        folder,
        'factories',
        """
        from sage.all__sagemath_symbolics import function as make
        LAG = make('lag', nargs=2)
        def build():
            return make('lead', print_func=str), make('a')
        """,
    )
    expected = [f'{path}:3: function factory at module level', f'{path}:5: function factory print_func']
    return _expected('_factory_calls', _factory_calls(path), expected)


def check_unnatural_lines_are_named(folder: Path) -> list[str]:
    """Return a finding unless a dash, a semicolon, and a banned phrase in prose are named, and code is not."""
    (folder / 'page.md').write_text(
        textwrap.dedent(
            """
            # Page

            - A list item stays allowed, and so does `a - b` in inline code.
            The text runs on — and on.
            It is a test - nothing more; it is robust.

            ```python
            total = 1 - 2  # a dash in code
            ```
            """,
        ),
        encoding='utf-8',
    )
    example = '"""\n# Example\n\nThe text runs on - and on.\n"""\n\n# %%\n# Note that it ends.\n'
    _write_module(folder, 'plot_example', example)
    expected = [
        'page.md:5: dash: The text runs on — and on.',
        'page.md:6: semicolon, dash, robust: It is a test - nothing more; it is robust.',
        'plot_example.py:5: dash: The text runs on - and on.',
        'plot_example.py:8: note that: Note that it ends.',
    ]
    return _expected('_unnatural_lines', _unnatural_lines(folder, ('page.md', 'plot_example.py')), expected)


def check_private_names_are_named(folder: Path) -> list[str]:
    """Return a finding unless a call of a private hook is named, and a subclass that extends the hook is not."""
    code = textwrap.dedent(
        """
        class QuietElement(Element):
            def _get_laws(self, namespace):
                return [*super()._get_laws(namespace), *self._get_extra(namespace)]

            def _get_extra(self, namespace):
                return []


        system._get_laws(namespace)
        system._parts
        """,
    )
    return _expected('_private_uses', _private_uses(code), [(10, 'system._get_laws'), (11, 'system._parts')])


def check_generated_classes_finds_a_planted_signature(folder: Path) -> list[str]:
    """Return a finding unless the reader finds a planted `__signature__` and a planted class built with `type`."""
    (folder / 'planted.py').write_text(
        "vars(f)['__signature__'] = 1\nf.__signature__ = 2\nC = type('C', (object,), {})\nd = type(object())\n",
    )
    return _expected('_generated_classes', _generated_classes(folder), {'planted.py': [1, 2, 3]})


def check_part_names_are_named(folder: Path) -> list[str]:
    """Return a finding unless a class and a docstring that call a component a part are named, and kept meanings not."""
    path = _write_module(
        folder,
        'planted',
        """
        class DoublingPart(BaseComponent):
            def _get_laws(self, namespace):
                \"\"\"Return the parts of the system.\"\"\"
                return [real_part(namespace.z)]

            def _run(self, value):
                \"\"\"Set its real part.\"\"\"
        """,
    )
    page = folder / 'page.md'
    page.write_text('# Page\n\nThe label of every part of a system is unique.\nAn element takes part in a run.\n')
    expected = [
        f'{path.relative_to(folder)}:2: part names a component: DoublingPart',
        f'{path.relative_to(folder)}:4: part names a component: Return the parts of the system.',
        'page.md:3: part names a component: The label of every part of a system is unique.',
    ]
    return _expected('_parts_of', _parts_of(folder, [path], [], [page]), expected)


SELF_CHECKS = (
    check_misplaced_constants_are_named,
    check_constant_gaps_are_named,
    check_unowned_functions_are_named,
    check_nonverb_methods_are_named,
    check_factory_calls_are_named,
    check_unnatural_lines_are_named,
    check_private_names_are_named,
    check_generated_classes_finds_a_planted_signature,
    check_part_names_are_named,
)
CHECKS = (
    check_tests_mirror_the_sources,
    check_no_source_hides_a_finding,
    check_documents_import_systems_from_their_subfield,
    check_documents_name_only_subfield_packages,
    check_readme_custom_system_runs_from_the_core_alone,
    check_documents_read_naturally,
    check_core_components_pages_cover_every_class,
    check_troubleshooting_covers_every_error,
    check_documents_show_only_public_names,
    check_core_never_imports_the_systems,
    check_shared_packages_hold_only_subpackages,
    check_core_components_obey_the_placement_rule,
    check_module_constants_lead_their_module,
    check_constant_blocks_hold_no_gap,
    check_classes_own_the_functions_they_read,
    check_methods_begin_with_a_verb,
    check_symbolic_functions_are_builtin_subclasses,
    check_core_names_carry_no_field_term,
    check_no_package_exports_a_molecular_name,
    check_no_class_defines_magic_methods,
    check_no_module_calls_a_magic_method,
    check_docstring_summaries_are_short,
    check_docstrings_read_naturally,
    check_class_summaries_name_their_kind,
    check_docstrings_say_main_public_methods,
    check_components_compare_and_name_uses,
    check_every_error_is_raised_in_the_library,
    check_every_dataclass_is_a_frozen_spec_or_result,
    check_enums_and_protocols_live_in_their_modules,
    check_every_base_derives_from_the_root,
    check_library_generates_no_classes_of_components,
    check_components_are_not_called_parts,
    check_no_module_serves_one_system,
    check_no_system_module_imports_another,
    check_every_general_module_serves_two_fields,
    check_connectors_live_in_the_library,
)


def run_checks(root: Path) -> list[str]:
    """Return the findings of the self-checks, each in its own folder, and then those of the checks of the root."""
    found = []
    for self_check in SELF_CHECKS:
        with tempfile.TemporaryDirectory() as folder:
            found += self_check(Path(folder))
    for check in CHECKS:
        found += check(root)
    return found


def main() -> int:
    """Run the checks from the root of the repository, print every finding, and return the exit status."""
    root = Path.cwd()
    if not (root / SCRIPT).is_file():
        sys.stderr.write(f'Run {SCRIPT} from the root of the repository.\n')
        return 2
    found = run_checks(root)
    sys.stdout.write(''.join(f'{line}\n' for line in found))
    sys.stdout.write(f'{len(found)} findings.\n')
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
