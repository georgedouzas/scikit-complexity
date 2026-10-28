"""Test the `RewritingCompiler` class."""

import time
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Annotated

import numpy as np
import pytest
import shapely

from skcomplex.analysis import ScalingAnalysis
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import BaseLaw
from skcomplex.core.system import System
from skcomplex.engine._utils import symbol
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import IncompatibleLawsError, ValidationError
from skcomplex.models.biology.ecology import PopulationElement
from skcomplex.models.mathematics.fractals import (
    AffineSpec,
    GeometrySpec,
    Letters,
    ModuleSpec,
    PatchElement,
    PatchStructureSpec,
    ProductionLaw,
    PrototileSpec,
    RewritingCompiler,
    RewritingError,
    SubstitutionLaw,
    WordElement,
    WordStructureSpec,
)
from skcomplex.models.mathematics.fractals._rewriting_problem_spec import REWRITING
from skcomplex.specs import ParameterSpec, StateSpec, StructureStateSpec
from skcomplex.types import FormKind

X = symbol('x')


class GrowthElement(WordElement):
    """Word with the parametric production `A(x) → F(x) A(x·R)` while `x > 0.1`."""

    def __init__(
        self,
        label: str,
        axiom: Annotated[Letters, StateSpec(domain=None)] = '',
        *,
        alphabet: str | Mapping[str, int] | None = None,
        R: Annotated[float, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, axiom, alphabet=alphabet)
        self.R = R

    def _get_laws(self, s):
        """Return the production."""
        production = ProductionLaw(
            ModuleSpec('A', (X,)),
            [ModuleSpec('F', (X,)), ModuleSpec('A', (X * s.R,))],
            condition=X > 0.1,
        )
        return [production]


class AlgaeElement(WordElement):
    """Word with the productions `A → AB` and `B → A` of the algae of Lindenmayer."""

    def _get_laws(self, s):
        """Return the productions."""
        return [ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]


class RuledWordElement(WordElement):
    """Word rewritten by the rules it takes."""

    def __init__(
        self,
        label: str,
        axiom: Annotated[Letters, StateSpec(domain=None)] = '',
        *,
        alphabet: str | Mapping[str, int] | None = None,
        ignore: str | Sequence[str] = '',
        geometry: GeometrySpec | None = None,
        max_size: int = 10_000_000,
        rules: Sequence[BaseLaw] = (),
    ) -> None:
        super().__init__(label, axiom, alphabet=alphabet, ignore=ignore, geometry=geometry, max_size=max_size)
        self.rules = rules

    def _get_laws(self, s):
        """Return the rules."""
        return list(self.rules)


class RuledPatchElement(PatchElement):
    """Patch rewritten by the rules it takes."""

    def __init__(
        self,
        label: str,
        *,
        prototiles: Sequence[PrototileSpec] = (),
        rules: Sequence[BaseLaw] = (),
    ) -> None:
        super().__init__(label, prototiles=prototiles)
        self.rules = rules

    def _get_laws(self, s):
        """Return the rules."""
        return list(self.rules)


def _fibonacci():
    """Return the L-system of the algae of Lindenmayer."""
    return System(
        elements=[RuledWordElement('word', axiom='A', rules=[ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')])],
    )


def test_generations_as_steps():
    """Test that an L-system compiles to the rewriting form and rewrites once per frame."""
    system = _fibonacci()
    assert compile_system(system, compiler=RewritingCompiler()).form == REWRITING
    trajectory = Simulation(system, compiler=RewritingCompiler()).check().run(until=5).trajectory_
    assert trajectory.times.tolist() == [0, 1, 2, 3, 4, 5]
    assert trajectory.values('word.size').tolist() == [1, 2, 3, 5, 8, 13]


def test_fractional_generation():
    """Test that a fractional step is refused."""
    with pytest.raises(ValidationError, match=r'dt = 0\.5 must be a positive integer'):
        Simulation(_fibonacci(), dt=0.5, compiler=RewritingCompiler()).check().run(until=2)


def test_parikh_vector():
    """Test that the symbol counts are consecutive Fibonacci numbers whose sum is the size."""
    trajectory = Simulation(_fibonacci(), compiler=RewritingCompiler()).check().run(until=10).trajectory_
    a, b = trajectory.values('word.A'), trajectory.values('word.B')
    assert np.array_equal(a[:-1], b[1:])
    assert np.array_equal(a[2:], a[1:-1] + a[:-2])
    assert np.array_equal(trajectory.values('word.size'), a + b)


def test_stochastic_choice():
    """Test that stochastic productions are chosen by their probabilities and reproduced by the seed."""
    system = System(
        elements=[
            RuledWordElement(
                'word',
                axiom='F' * 10000,
                alphabet='F+',
                rules=[ProductionLaw('F', 'F+F', probability=0.3), ProductionLaw('F', 'F', probability=0.7)],
            ),
        ],
    )
    first = Simulation(system, random_state=3, compiler=RewritingCompiler()).check().run(until=1).trajectory_
    second = Simulation(system, random_state=3, compiler=RewritingCompiler()).check().run(until=1).trajectory_
    share = first.structure('word', 1).count('+') / 10000
    assert share == pytest.approx(0.3, abs=0.015)
    assert np.array_equal(first.structure('word', 1).codes, second.structure('word', 1).codes)


def test_probabilities_that_do_not_sum_to_one():
    """Test that stochastic probabilities that sum to 0.9 are refused with the symbol and the sum."""
    system = System(
        elements=[
            RuledWordElement(
                'word',
                axiom='F',
                rules=[ProductionLaw('F', 'FF', probability=0.4), ProductionLaw('F', 'F', probability=0.5)],
            ),
        ],
    )
    with pytest.raises(RewritingError, match=r"of 'F' in word 'word' have probabilities that sum to 0\.9"):
        compile_system(system, compiler=RewritingCompiler())


def test_context_with_ignored_symbols():
    """Test a left context that skips an ignored symbol."""
    system = System(
        elements=[RuledWordElement('word', axiom='BAA+A', ignore='+', rules=[ProductionLaw('A', 'B', left='B')])],
    )
    trajectory = Simulation(system, compiler=RewritingCompiler()).check().run(until=3).trajectory_
    assert [trajectory.structure('word', frame).text for frame in range(1, 4)] == ['BBA+A', 'BBB+A', 'BBB+B']


def test_bracketed_contexts():
    """Test that contexts follow branches: a branch sees its parent on the left, and the right skips branches."""
    system = System(
        elements=[
            RuledWordElement(
                'word',
                axiom='AB[C]D[E]',
                rules=[
                    ProductionLaw('C', 'X', left='B'),
                    ProductionLaw('B', 'Y', right='D'),
                    ProductionLaw('D', 'Z', left='B'),
                    ProductionLaw('E', 'W', right='F'),
                ],
            ),
        ],
    )
    trajectory = Simulation(system, compiler=RewritingCompiler()).check().run(until=1).trajectory_
    assert trajectory.structure('word', 1).text == 'AY[X]Z[E]'


def test_context_takes_precedence():
    """Test that a context-sensitive production wins over a context-free one of the same symbol."""
    system = System(
        elements=[
            RuledWordElement('word', axiom='BAA', rules=[ProductionLaw('A', 'C'), ProductionLaw('A', 'B', left='B')]),
        ],
    )
    assert (
        Simulation(system, compiler=RewritingCompiler()).check().run(until=1).trajectory_.structure('word', 1).text
        == 'BBC'
    )


def test_parametric_growth():
    """Test parametric productions with a condition and successor parameters of a system parameter."""
    word = GrowthElement('plant', axiom=[ModuleSpec('A', (1.0,))], alphabet={'A': 1, 'F': 1})
    system = System(elements=[word])
    simulation = Simulation(system, compiler=RewritingCompiler()).check().run(until=6)
    trajectory = simulation.trajectory_
    assert trajectory.structure('plant', 5).text == 'F(1) F(0.5) F(0.25) F(0.125) A(0.0625)'
    assert trajectory.structure('plant', 6).text == trajectory.structure('plant', 5).text
    assert trajectory.values('plant.R')[0] == 0.5
    assert trajectory.structure('plant', 1).modules()[1] == ModuleSpec('A', (0.5,))


def test_parameter_change_between_generations():
    """Test that a production parameter changed by a schedule applies from the next generation."""
    word = GrowthElement('plant', axiom=[ModuleSpec('A', (1.0,))], alphabet={'A': 1, 'F': 1})
    trajectory = (
        Simulation(System(elements=[word]), schedule={'plant.R': lambda t: 0.25}, compiler=RewritingCompiler())
        .check()
        .run(until=1)
        .trajectory_
    )
    assert trajectory.structure('plant', 1).text == 'F(1) A(0.25)'


def test_exponential_growth_stopped():
    """Test that the size guard raises before the generation that exceeds it and keeps the earlier frames."""
    system = System(
        elements=[RuledWordElement('koch', axiom='F', max_size=10**5, rules=[ProductionLaw('F', 'F+F--F+F')])],
    )
    simulation = Simulation(system, compiler=RewritingCompiler()).check()
    with pytest.raises(RewritingError, match="Word 'koch' would reach 152916 symbols at generation 8"):
        simulation.run(until=10)
    assert len(simulation.trajectory_.times) == 8


@pytest.mark.benchmark
def test_benchmark_of_a_large_word():
    """Test that one generation of a word of a million symbols takes under 0.1 s."""
    problem = compile_system(
        System(elements=[RuledWordElement('koch', axiom='F', alphabet='F+-', rules=[ProductionLaw('F', 'F+F--F+F')])]),
        compiler=RewritingCompiler(),
    )
    compiled = problem.rewriting[0]
    word = WordStructureSpec(tuple(compiled.names), np.zeros(10**6, dtype=np.int64), np.zeros((10**6, 0)), 0)
    start = time.perf_counter()
    rewritten = compiled._rewrite(word, problem.initial_parameters, np.random.default_rng(0))
    assert time.perf_counter() - start < 0.1
    assert len(rewritten) == 8 * 10**6


def test_malformed_systems():
    """Test the errors of unknown symbols, non-symbol predecessor parameters, mixed laws, and wrong targets."""
    with pytest.raises(RewritingError, match='not in its alphabet'):
        compile_system(
            System(elements=[RuledWordElement('word', axiom='A', alphabet='A', rules=[ProductionLaw('A', 'AB')])]),
            compiler=RewritingCompiler(),
        )
    with pytest.raises(RewritingError, match='distinct symbols'):
        compile_system(
            System(
                elements=[
                    RuledWordElement(
                        'word',
                        axiom=[ModuleSpec('A', (1.0,))],
                        rules=[ProductionLaw(ModuleSpec('A', (1.0,)), 'A')],
                    ),
                ],
            ),
            compiler=RewritingCompiler(),
        )
    with pytest.raises(RewritingError, match='neither a parameter'):
        compile_system(
            System(
                elements=[
                    RuledWordElement(
                        'word',
                        axiom=[ModuleSpec('A', (1.0,))],
                        rules=[ProductionLaw(ModuleSpec('A', (X,)), [ModuleSpec('A', (X * symbol('q'),))])],
                    ),
                ],
            ),
            compiler=RewritingCompiler(),
        )
    with pytest.raises(IncompatibleLawsError, match='takes no part in rewriting'):
        compile_system(
            System(elements=[WordElement('word', axiom='A'), PopulationElement('rabbits')]),
            compiler=RewritingCompiler(),
        )
    with pytest.raises(IncompatibleLawsError, match='takes no part in rewriting'):
        compile_system(
            System(elements=[WordElement('word', axiom='A')], interactions=[Interaction('rules')]),
            compiler=RewritingCompiler(),
        )
    with pytest.raises(RewritingError, match='rewritten by ProductionLaws'):
        compile_system(
            System(elements=[RuledWordElement('word', axiom='A', rules=[SubstitutionLaw('A', [])])]),
            compiler=RewritingCompiler(),
        )


def test_koch_curve_length():
    """Test the word geometry of the Koch L-system at generation 4."""
    system = System(
        elements=[
            RuledWordElement(
                'koch',
                axiom='F',
                geometry=GeometrySpec(angle=60.0, contraction=1 / 3),
                rules=[ProductionLaw('F', 'F+F--F+F')],
            ),
        ],
    )
    trajectory = Simulation(system, compiler=RewritingCompiler()).check().run(until=4).trajectory_
    curve = trajectory.structure('koch', 4)
    segments = curve.segments()
    assert segments.shape == (256, 2, 2)
    assert curve.length() == pytest.approx((4 / 3) ** 4, abs=1e-12)
    assert np.allclose(segments[0, 0], [0, 0], atol=1e-12)
    assert np.allclose(segments[-1, 1], [1, 0], atol=1e-12)
    assert trajectory.values('koch.length')[-1] == pytest.approx((4 / 3) ** 4, abs=1e-12)


def test_branches_return():
    """Test that a popped branch returns the cursor to where the branch began."""
    curve = (
        Simulation(
            System(elements=[WordElement('plant', axiom='F[+F]F', geometry=GeometrySpec())]),
            compiler=RewritingCompiler(),
        )
        .check()
        .run(0)
        .trajectory_
    )
    segments = curve.structure('plant', 0).segments()
    assert len(segments) == 3
    assert np.allclose(segments[2, 0], segments[1, 0])
    assert np.allclose(segments[1, 1], [1, 1])
    assert curve.structure('plant', 0).depths.tolist() == [0, 1, 0]


def test_three_dimensional_word_geometry():
    """Test that pitching up and rolling move the cursor out of the plane."""
    geometry = GeometrySpec(angle=90.0, dimension=3)
    curve = (
        Simulation(
            System(elements=[WordElement('plant', axiom='F&F/+F', geometry=geometry)]),
            compiler=RewritingCompiler(),
        )
        .check()
        .run(0)
        .trajectory_
    )
    segments = curve.structure('plant', 0).segments()
    assert segments.shape == (3, 2, 3)
    assert np.allclose(segments[1, 1], [1, 0, -1])
    assert curve.structure('plant', 0).length() == pytest.approx(3.0)


def test_module_parameters_override_the_word_geometry():
    """Test that a module's first parameter overrides the step of a move and the angle of a turn."""
    word = WordElement(
        'plant',
        axiom=[ModuleSpec('F', (2.0,)), ModuleSpec('+', (90.0,)), ModuleSpec('F', (1.0,))],
        geometry=GeometrySpec(angle=10.0),
    )
    segments = (
        Simulation(System(elements=[word]), compiler=RewritingCompiler())
        .check()
        .run(0)
        .trajectory_.structure('plant', 0)
        .segments()
    )
    assert np.allclose(segments[1], [[2, 0], [2, 1]])


def test_word_geometry_errors():
    """Test that unknown actions and dimensions are refused."""
    with pytest.raises(RewritingError, match='has no action'):
        GeometrySpec(symbols={'G': 'jump'}).check()
    with pytest.raises(RewritingError, match='dimension 4'):
        GeometrySpec(dimension=4).check()


def _cantor(generations):
    """Return the trajectory of the Cantor substitution."""
    interval = PrototileSpec('interval', [(0, 0), (1, 0)])
    rule = SubstitutionLaw(
        'interval',
        [('interval', AffineSpec(scale=1 / 3)), ('interval', AffineSpec(scale=1 / 3, translation=(2 / 3, 0)))],
    )
    return (
        Simulation(
            System(elements=[RuledPatchElement('cantor', prototiles=[interval], rules=[rule])]),
            compiler=RewritingCompiler(),
        )
        .check()
        .run(until=generations)
        .trajectory_
    )


def test_cantor_intervals():
    """Test the intervals of the Cantor substitution after 5 generations."""
    patch = _cantor(5).structure('cantor', 5)
    assert patch.parts == 32
    assert patch.length() == pytest.approx((2 / 3) ** 5, abs=1e-12)
    lefts = np.sort(patch.segments()[:, 0, 0])
    expected = sorted(
        sum(digit * 3.0 ** -(place + 1) for place, digit in enumerate(digits)) for digits in np.ndindex(*(2,) * 5)
    )
    assert np.allclose(lefts, [value * 2 for value in expected], atol=1e-12)


def test_structure_of_a_generation():
    """Test that a recorded structure is a copy that later generations do not change."""
    trajectory = _cantor(4)
    patch = trajectory.structure('cantor', 2)
    assert isinstance(patch, PatchStructureSpec)
    assert patch.parts == 4
    assert not patch.matrices.flags.writeable
    assert patch.length() == pytest.approx(4 / 9)


def test_unknown_element():
    """Test that asking for a structure that is not a rewritten element lists the rewritten elements."""
    with pytest.raises(KeyError, match='The rewritten elements of the system are: cantor'):
        _cantor(1).structure('other', 0)


def _sierpinski(generations):
    """Return the trajectory of the Sierpiński triangle."""
    height = np.sqrt(3) / 2
    triangle = PrototileSpec('triangle', [(0, 0), (1, 0), (0.5, height)])
    corners = [(0, 0), (0.5, 0), (0.25, height / 2)]
    rule = SubstitutionLaw('triangle', [('triangle', AffineSpec(scale=0.5, translation=corner)) for corner in corners])
    return (
        Simulation(
            System(elements=[RuledPatchElement('gasket', prototiles=[triangle], rules=[rule])]),
            compiler=RewritingCompiler(),
        )
        .check()
        .run(until=generations)
        .trajectory_
    )


def test_sierpinski_triangle_area():
    """Test the area of the Sierpiński triangle after 6 generations."""
    trajectory = _sierpinski(6)
    patch = trajectory.structure('gasket', 6)
    assert patch.parts == 729
    assert patch.area() == pytest.approx(0.75**6 * np.sqrt(3) / 4, abs=1e-12)
    assert trajectory.values('gasket.triangle')[-1] == 729


def test_gasket_box_counting():
    """Test the box-counting dimension of the raster of the Sierpiński triangle at generation 8."""
    grid = _sierpinski(8).structure('gasket', 8).raster(1024)
    boxes = ScalingAnalysis('box-counting', scales=[4, 8, 16, 32, 64, 128]).check().prepare(grid).run()
    assert boxes.exponent_ == pytest.approx(np.log(3) / np.log(2), abs=0.05)


def test_measures_and_raster_errors():
    """Test the bounds, the number of parts, and the refused resolutions."""
    patch = _sierpinski(1).structure('gasket', 1)
    assert patch.parts == 3
    assert patch.bounds() == pytest.approx((0, 0, 1, np.sqrt(3) / 2))
    with pytest.raises(ValidationError, match='positive integer'):
        patch.raster(0)


def test_chair_tiling_covers_its_chair():
    """Test that the chair substitution tiles the chair it starts from without overlap."""
    chair = PrototileSpec('chair', [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    pieces = [
        ('chair', AffineSpec(scale=0.5)),
        ('chair', AffineSpec(scale=0.5, translation=(0.5, 0.5))),
        ('chair', AffineSpec(scale=0.5, rotation=90, translation=(2, 0))),
        ('chair', AffineSpec(scale=0.5, rotation=-90, translation=(0, 2))),
    ]
    trajectory = (
        Simulation(
            System(
                elements=[RuledPatchElement('chairs', prototiles=[chair], rules=[SubstitutionLaw('chair', pieces)])],
            ),
            compiler=RewritingCompiler(),
        )
        .check()
        .run(3)
        .trajectory_
    )
    patch = trajectory.structure('chairs', 3)
    assert patch.parts == 64
    union = shapely.union_all(patch.geometry())
    assert shapely.symmetric_difference(union, shapely.Polygon(chair.vertices)).area == pytest.approx(0.0, abs=1e-9)
    assert patch.area() == pytest.approx(3.0, abs=1e-9)


def test_unsubstituted_tiles_are_kept_and_named_errors():
    """Test that tiles without substitution are kept, and unknown prototiles are refused."""
    square = PrototileSpec('square', [(0, 0), (1, 0), (1, 1), (0, 1)])
    dot = PrototileSpec('dot', [(0, 0), (0.1, 0)])
    rule = SubstitutionLaw('square', [('square', AffineSpec(scale=0.5)), ('dot', AffineSpec())])
    patch = RuledPatchElement('patch', prototiles=[square, dot], rules=[rule])
    trajectory = Simulation(System(elements=[patch]), compiler=RewritingCompiler()).check().run(until=2).trajectory_
    assert trajectory.values('patch.dot').tolist() == [0, 1, 2]
    with pytest.raises(RewritingError, match='not one of'):
        compile_system(
            System(
                elements=[RuledPatchElement('patch', prototiles=[square, dot], rules=[SubstitutionLaw('circle', [])])],
            ),
            compiler=RewritingCompiler(),
        )
    with pytest.raises(RewritingError, match='two substitutions'):
        compile_system(
            System(elements=[RuledPatchElement('patch', prototiles=[square, dot], rules=[rule, rule])]),
            compiler=RewritingCompiler(),
        )
    with pytest.raises(RewritingError, match='at least one prototile'):
        compile_system(System(elements=[RuledPatchElement('empty')]), compiler=RewritingCompiler())


def test_rewriting_compiler_reports_a_classical_system():
    """Test that the rewriting compiler accepts a system of words and reports a classical one."""
    compiler = RewritingCompiler().check()
    assert compiler.prepare(System(elements=[WordElement('algae', axiom='A')])).report_.ok
    assert not compiler.prepare(System(elements=[PopulationElement('rabbits', n=10)])).report_.ok


def test_rewriting_compiler_runs_the_problem():
    """Test that the prepared rewriting compiler returns the rewriting problem when run."""
    words = System(elements=[WordElement('algae', axiom='A')])
    problem = RewritingCompiler().check().prepare(words).run()
    assert problem.form == REWRITING


def test_snapshot_is_a_structure_state():
    """Test that the rewriter hands its structures and measures to the simulation as a structure state."""
    snapshot = (
        Simulation(System(elements=[AlgaeElement('algae', axiom='A')]), compiler=RewritingCompiler())
        .check()
        .run(until=2)
        .stepper_.snapshot_
    )
    assert isinstance(snapshot, StructureStateSpec)
    assert snapshot.structures['algae'].parts == 3


def test_rule_of_a_word():
    """Test that a word rewrites itself by the productions it returns."""
    trajectory = (
        Simulation(System(elements=[AlgaeElement('algae', axiom='A')]), compiler=RewritingCompiler())
        .check()
        .run(until=5)
        .trajectory_
    )
    assert trajectory.structure('algae', 5).text == 'ABAABABAABAAB'


def test_the_library_names_no_rewriting():
    """Test that the core, the engine, the specs, and the types name none of the rewriting classes or its form."""
    names = ('WordElement', 'PatchElement', 'ProductionLaw', 'SubstitutionLaw', 'RewritingCompiler', 'RewritingSolver')
    root = Path(str(files('skcomplex')))
    sources = [*root.joinpath('core').rglob('*.py'), *root.joinpath('engine').rglob('*.py')]
    sources += [*root.joinpath('specs').rglob('*.py'), root / 'types.py']
    found = [(path.name, name) for path in sources for name in names if name in path.read_text()]
    assert found == []
    assert 'REWRITING' not in FormKind.__members__
