"""Test the `base` module of `skcomplex.core.components`."""

import ast
import importlib
import inspect
import pkgutil
import re
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import pytest
from sklearn.base import clone

from skcomplex.core.components.base import BasePart
from skcomplex.core.components.elements import BaseElement, Element
from skcomplex.core.components.interactions import BaseInteraction
from skcomplex.core.components.laws import BaseLaw
from skcomplex.core.components.spaces import BaseNeighborhood, BaseSpace, BaseTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec

SOURCE = str(Path(str(files('skcomplex'))))
PACKAGES = (
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
SUMMARY_LENGTH = 80
KINDS = {
    'Element': 'element',
    'Interaction': 'interaction',
    'Law': 'law',
    'Space': 'space',
    'Time': 'time',
    'Neighborhood': 'neighbourhood',
    'System': 'system',
    'Solver': 'solver',
    'Backend': 'backend',
    'Observer': 'observer',
    'Spec': 'spec',
    'Result': 'result',
    'Pass': 'pass',
    'Error': 'error',
}
FAMILIES = tuple(
    f'skcomplex.core.components.{family}' for family in ('elements', 'interactions', 'laws', 'spaces', 'systems')
)
REWRITTEN: tuple[str, ...] = ('skcomplex.core.components.elements',)


class DoublingPart(BasePart):
    """Part that doubles its input after counting its preparations."""

    def __init__(self, label, factor=2):
        super().__init__(label)
        self.factor = factor

    def _prepare(self, context=None):
        """Check the factor, then count the preparation and keep the context."""
        super()._prepare(context)
        if not isinstance(self.factor, int | float):
            error_msg = f'The factor {self.factor!r} is not a number.'
            raise ValidationError(error_msg)
        self.prepared_ = getattr(self, 'prepared_', 0) + 1
        self.context_ = context

    def _get_laws(self, namespace):
        """Return no laws."""
        return []

    def _run(self, inputs):
        """Set `doubled_`, the input times the factor."""
        self.doubled_ = self.factor * inputs


class SpecifiedPart(BasePart):
    """Part that declares a state, a parameter, and a role, and one plain argument."""

    def __init__(
        self,
        label,
        x: Annotated[float, StateSpec(order=2)] = 0.0,
        k: Annotated[float, ParameterSpec()] = 1.0,
        target: Annotated[str, RoleSpec()] = 'a',
        plain=None,
    ):
        super().__init__(label)
        self.x = x
        self.k = k
        self.target = target
        self.plain = plain

    def _prepare(self, context=None):
        """Check the label and the declared specs, and prepare nothing else."""
        super()._prepare(context)

    def _get_laws(self, namespace):
        """Return no laws."""
        return []

    def _run(self, inputs):
        """Run nothing."""


class ExtendedPart(SpecifiedPart):
    """Part that takes the state of its base again without annotating it, and adds a parameter first."""

    def __init__(self, label, r: Annotated[float, ParameterSpec()] = 0.5, x=1.0):
        super().__init__(label, x=x)
        self.r = r


class InvalidSpecPart(SpecifiedPart):
    """Part that declares a state of an invalid order."""

    def __init__(self, label, y: Annotated[float, StateSpec(order=0)] = 0.0):
        super().__init__(label)
        self.y = y


def _modules():
    """Return every module of the packages that define the library's classes."""
    for name in PACKAGES:
        package = importlib.import_module(name)
        yield package
        for info in pkgutil.walk_packages(package.__path__, f'{name}.'):
            yield importlib.import_module(info.name)


def _hand_written(cls):
    """Return the magic methods a class defines itself in source code, leaving out those Python generates.

    Dataclasses generate methods, and from Python 3.14 the annotations of a class compile to `__annotate_func__`.
    """
    names = []
    for name, value in vars(cls).items():
        if not (name.startswith('__') and name.endswith('__')) or not callable(value) or name == '__annotate_func__':
            continue
        code = getattr(value, '__code__', None)
        if code is not None and code.co_filename.startswith(SOURCE):
            names.append(name)
    return names


def _core_modules():
    """Return the dotted name and the syntax tree of every module of `skcomplex.core`, read from its source."""
    for path in sorted((Path(SOURCE) / 'core').rglob('*.py')):
        parts = path.relative_to(Path(SOURCE).parent).with_suffix('').parts
        module = '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)
        yield module, ast.parse(path.read_text(encoding='utf-8'))


def _docstrings():
    """Return the dotted name, the docstring, and the publicity of every docstring of `skcomplex.core`.

    The walk descends into classes, not into functions.
    """
    for module, tree in _core_modules():
        yield from _walk_docstrings(tree, module, public=True)


def _walk_docstrings(node, name, *, public):
    """Return the dotted name, the docstring, and the publicity of a node and of the definitions in its body."""
    docstring = ast.get_docstring(node)
    if docstring:
        yield name, docstring, public
    if isinstance(node, ast.Module | ast.ClassDef):
        for child in node.body:
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                is_public = public and not child.name.startswith('_')
                yield from _walk_docstrings(child, f'{name}.{child.name}', public=is_public)


def _rewritten(name):
    """Return whether a dotted name lies in a package whose docstrings are rewritten."""
    return any(name == prefix or name.startswith(f'{prefix}.') for prefix in REWRITTEN)


def _public_definitions(kind):
    """Return the dotted name, the name, and the docstring of every top-level public class or function rewritten."""
    for module, tree in _core_modules():
        for node in tree.body:
            docstring = ast.get_docstring(node) if isinstance(node, kind) else None
            if docstring and not node.name.startswith('_') and _rewritten(module):
                yield f'{module}.{node.name}', node.name, docstring


def _prose_lines(docstring):
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


def _prose_faults(line):
    """Return the semicolons, dashes used as punctuation, and banned phrases of a prose line."""
    faults = []
    if ';' in line:
        faults.append('semicolon')
    if re.search('[\u2013\u2014]', line) or ' - ' in line[1:]:
        faults.append('dash')
    lowered = line.lower()
    faults += [phrase for phrase in BANNED_PHRASES if re.search(rf'(?<![\w-]){re.escape(phrase)}(?![\w-])', lowered)]
    return faults


def test_prepare_and_run():
    """Test that preparing derives the attributes of the part from a context, and running acts."""
    part = DoublingPart('doubling').prepare('context')
    assert (part.prepared_, part.context_) == (1, 'context')
    assert part.run(3).doubled_ == 6


def test_prepare_checks_the_label_and_the_arguments():
    """Test that preparing rejects a label that is not a string and an argument the part cannot take."""
    with pytest.raises(ValidationError, match='label of DoublingPart must be a string'):
        DoublingPart(1).prepare()
    with pytest.raises(ValidationError, match='not a number'):
        DoublingPart('doubling', factor='two').prepare()


def test_prepare_checks_every_declared_spec():
    """Test that preparing a part checks the spec of every annotated argument."""
    with pytest.raises(ValidationError, match='order'):
        InvalidSpecPart('invalid').prepare()


def test_get_specs_returns_the_annotated_arguments_in_constructor_order():
    """Test that the specs of a part are the annotated arguments of its constructor, in order, without plain ones."""
    specs = SpecifiedPart._get_specs()
    assert list(specs) == ['x', 'k', 'target']
    assert specs['x'] == StateSpec(order=2)
    assert isinstance(specs['target'], RoleSpec)


def test_get_specs_keeps_the_specs_of_the_bases():
    """Test that a subclass keeps the spec of an argument of its base that it takes again without annotating it."""
    specs = ExtendedPart._get_specs()
    assert list(specs) == ['r', 'x']
    assert specs['x'] == StateSpec(order=2)


def test_constructor_only_stores_its_arguments():
    """Test that a part stores its arguments as given, so scikit-learn lists and clones them."""
    part = SpecifiedPart('a', x=[1.0], plain='plain')
    assert part.get_params() == {'k': 1.0, 'label': 'a', 'plain': 'plain', 'target': 'a', 'x': [1.0]}
    cloned = clone(part)
    assert cloned.get_params() == part.get_params()
    assert cloned is not part


def test_element_runs_into_its_laws():
    """Test that running an element sets its laws for the namespace, as interactions and systems do."""

    class StillElement(Element):
        def _get_laws(self, namespace):
            return []

    assert StillElement('a').prepare().run(SimpleNamespace()).laws_ == []


def test_parts_derive_from_the_part_base():
    """Test that every part of a system is a part, and a part cannot be made without its hooks."""
    parts = (BaseElement, BaseInteraction, BaseLaw, CompositeSystem, BaseSpace, BaseTime, BaseNeighborhood)
    assert all(issubclass(part, BasePart) for part in parts)
    with pytest.raises(TypeError, match='abstract'):
        BasePart('part')


def test_core_names_carry_no_field_term():
    """Test that no public name of a package of `skcomplex.core.components` contains a term of a scientific field."""
    found = {}
    for module in _modules():
        if not module.__name__.startswith('skcomplex.core.components'):
            continue
        for name in getattr(module, '__all__', []):
            words = re.findall('[A-Z][a-z]*', name)
            terms = sorted(set(words) & set(FIELD_TERMS))
            if terms:
                found[f'{module.__name__}.{name}'] = terms
    assert found == {}, found


def test_no_package_exports_a_molecular_name():
    """Test that the core, the engine, and the analysis export no name of RNA, proteins, or hydrophobicity."""
    terms = ('rna', 'protein', 'genotype', 'hydrophobicity', 'residue', 'nucleotide')
    found = [
        f'{module.__name__}.{name}'
        for module in _modules()
        if module.__name__.startswith(('skcomplex.core.components', 'skcomplex.core.engine', 'skcomplex.core.analysis'))
        for name in getattr(module, '__all__', [])
        if any(term in re.sub(r'[^a-z]', ' ', re.sub(r'([A-Z])', r' \1', name).lower()).split() for term in terms)
    ]
    assert found == []


def test_no_class_defines_magic_methods():
    """Test that no class of the library defines a magic method but `__init__`."""
    found = {}
    for module in _modules():
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            extra = set(_hand_written(cls)) - {'__init__'}
            if extra:
                found[f'{module.__name__}.{cls.__qualname__}'] = sorted(extra)
    assert found == {}, found


def test_no_module_calls_a_magic_method():
    """Test that no module of the library reads a magic method such as `__new__` or `__getitem__` of an object."""
    allowed = {
        '__init__',
        '__name__',
        '__qualname__',
        '__doc__',
        '__module__',
        '__mro__',
        '__version__',
        '__subclasses__',
    }
    found = []
    for path in Path(SOURCE).rglob('*.py'):
        names = re.findall(r'\.(__[a-z_]+__)\b', path.read_text(encoding='utf-8'))
        found += [f'{path.name}: {name}' for name in names if name not in allowed]
    assert found == [], found


def test_docstring_summaries_are_short():
    """Test that the summary line of every docstring of `skcomplex.core` is at most 80 characters long."""
    found = {
        name: len(docstring.splitlines()[0])
        for name, docstring, _ in _docstrings()
        if len(docstring.splitlines()[0]) > SUMMARY_LENGTH
    }
    assert found == {}, found


def test_docstrings_read_naturally():
    """Test that no docstring of the core holds a semicolon, and no public one a dash or a banned phrase."""
    found = {}
    for name, docstring, public in _docstrings():
        for line in _prose_lines(docstring):
            faults = _prose_faults(line) if public else ['semicolon'] * (';' in line)
            if faults:
                found.setdefault(name, []).append((line, faults))
    assert found == {}, found


def test_class_summaries_name_their_kind():
    """Test that a rewritten public class names its kind after an article, and no summary holds a field term."""
    found = {}
    for name, short, docstring in _public_definitions(ast.ClassDef):
        summary = docstring.splitlines()[0]
        kind = next((word for suffix, word in KINDS.items() if short.endswith(suffix)), None)
        if not summary.startswith(('A ', 'An ')) or (kind is not None and kind not in summary.lower()):
            found[name] = summary
    for name, _, docstring in [*_public_definitions(ast.ClassDef), *_public_definitions(ast.FunctionDef)]:
        words = {word.lower() for word in re.findall('[A-Za-z]+', docstring.splitlines()[0])}
        terms = sorted(term for term in FIELD_TERMS if term.lower() in words)
        if terms:
            found[name] = terms
    assert found == {}, found


def test_docstrings_say_main_public_methods():
    """Test that no rewritten docstring calls the main public methods verbs."""
    found = [
        name
        for name, docstring, _ in _docstrings()
        if _rewritten(name) and re.search(r'\bverbs?\b', docstring, flags=re.IGNORECASE)
    ]
    assert found == [], found


def test_components_compare_and_name_uses():
    """Test that every rewritten public class of a family compares itself with its nearest part and names its uses."""
    found = []
    for name, _, docstring in _public_definitions(ast.ClassDef):
        if not name.startswith(FAMILIES):
            continue
        paragraphs = [' '.join(paragraph.split()) for paragraph in docstring.split('\n\n')]
        compares = any(p.startswith('Unlike') and 'Choose it when' in p for p in paragraphs)
        names_uses = any(p.startswith('It appears in') for p in paragraphs)
        if not (compares and names_uses):
            found.append(name)
    assert found == [], found
