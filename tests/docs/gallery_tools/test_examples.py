"""Test the scaffolding, the coverage, and the budget of the gallery examples."""

from pathlib import Path

import pytest
from gallery_tools import (
    SCAFFOLD_MARK,
    Budget,
    Coverage,
    GalleryError,
    check_budget,
    check_coverage,
    find_examples,
    gallery_interval,
    library_models,
    scaffold_example,
    scaffold_examples,
)

from skcomplex.models import describe

DOCS = Path(__file__).parent.parent.parent.parent / 'docs'


def _write(folder, field, name, model):
    """Write an example script that names a model."""
    path = folder / field / f'plot_{name}.py'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'"""\n# {name}\n"""\n\nMODEL = {model!r}\n')
    return path


def test_check_coverage_names_missing_model(tmp_path):
    """Test that a registered model without an example is named."""
    _write(tmp_path, 'biology/ecology', 'lotka_volterra', 'lotka_volterra')
    coverage = check_coverage(tmp_path, [describe('lotka_volterra'), describe('sir')])
    assert coverage.missing == ['sir']
    assert not coverage.complete


@pytest.mark.parametrize(
    ('field', 'name', 'model'),
    [
        ('biology/ecology', 'unknown', 'unknown_model'),
        ('chemistry/kinetics', 'lotka_volterra', 'lotka_volterra'),
    ],
)
def test_check_coverage_names_orphan_example(tmp_path, field, name, model):
    """Test that an example of an unregistered model or in the wrong folder is named."""
    _write(tmp_path, field, name, model)
    coverage = check_coverage(tmp_path, [describe('lotka_volterra')])
    assert coverage.orphans == [f'{field}/plot_{name}.py']


def test_check_coverage_names_duplicate_examples(tmp_path):
    """Test that two examples of one model are both named."""
    _write(tmp_path, 'biology/ecology', 'lotka_volterra', 'lotka_volterra')
    _write(tmp_path, 'biology/ecology', 'predation', 'lotka_volterra')
    coverage = check_coverage(tmp_path, [describe('lotka_volterra')])
    assert len(coverage.orphans) == 2


def test_check_coverage_of_repository_is_complete():
    """Test that every registered model has exactly one example in the documentation."""
    coverage = check_coverage(DOCS / 'examples')
    assert coverage.missing == []
    assert coverage.orphans == []


def test_find_examples_marks_scaffolds():
    """Test that the curated reference examples are not scaffolds."""
    examples = {example.model: example for example in find_examples(DOCS / 'examples')}
    assert len(examples) == len(library_models())
    for name in ('lotka_volterra', 'lorenz', 'gray_scott', 'game_of_life', 'wolf_sheep'):
        assert not examples[name].scaffolded


def test_scaffold_example_runs_model_with_blueprint():
    """Test that a scaffold names its model, sends the default blueprint, and seeds the run."""
    script = scaffold_example(describe('wolf_sheep'), 'Wolf and sheep')
    assert "MODEL = 'wolf_sheep'" in script
    assert "blueprint='default'" in script
    assert 'random_state=0)' in script
    assert SCAFFOLD_MARK in script
    assert max(len(line) for line in script.splitlines()) <= 120


def test_scaffold_examples_writes_missing_examples_and_headers(tmp_path):
    """Test that the generator writes every example and the headers of the fields and subfields."""
    written = scaffold_examples(tmp_path, DOCS / 'overview' / 'inventory.toml')
    assert len(written) == len(library_models())
    assert (tmp_path / 'biology' / 'README.md').exists()
    assert (tmp_path / 'biology' / 'ecology' / 'README.md').exists()
    assert scaffold_examples(tmp_path, DOCS / 'overview' / 'inventory.toml') == []


@pytest.mark.parametrize(
    ('until', 'dt', 'interval'),
    [(10.0, 1.0, 1), (50.0, 0.1, 2), (2000.0, 0.01, 800), (1000.0, 1.0, 4)],
)
def test_gallery_interval_keeps_frames_within_budget(until, dt, interval):
    """Test that the interval keeps at most 300 frames and divides the number of steps when it can."""
    assert gallery_interval(until, dt, 300) == interval
    assert round(until / dt) // interval + 1 <= 300


def test_check_budget_names_exceeded_limits():
    """Test that an example over the budget is named with each exceeded limit."""
    with pytest.raises(GalleryError, match=r'plot_x exceeds the budget: .*bytes.*seconds'):
        check_budget('plot_x', 10, 30_000_000, 90.0, Budget())
    check_budget('plot_x', 300, 20_000_000, 60.0, Budget())


def test_coverage_problems_name_models_without_examples_and_examples_without_screenshots():
    """Test that a strict build reports a new model without an example and an example without a screenshot."""
    assert Coverage().problems([]) == []
    problems = Coverage(missing=['new_model']).problems(['plot_unrun'])
    assert problems == [
        'These registered models have no gallery example: new_model.',
        'These gallery examples have no screenshot: plot_unrun.',
    ]
