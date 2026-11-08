"""Test the `_draw_recorder` module of `skcomplex.core.engine.problems`."""

import numpy as np
import pytest
from scipy import stats

from skcomplex.core.engine.problems._draw_recorder import DrawRecorder
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import DrawSpec

KICK = DrawSpec('normal', 'kick')


def test_recorder_keeps_values_by_name_and_step_with_their_arguments():
    """Test that recorded draws keep their steps, values, arguments, and log-probability."""
    recorder = DrawRecorder().prepare()
    for step, value in enumerate([0.5, -0.2], start=1):
        recorder.run(step).draw('walker.kick', KICK, [0.0, 2.0], np.array([value]))
    record = recorder.log_.draws['walker.kick']
    assert record.steps.tolist() == [1, 2]
    assert record.values.tolist() == [0.5, -0.2]
    assert record.arguments['deviation'].tolist() == [2.0, 2.0]
    assert recorder.log_probability_ == pytest.approx(stats.norm.logpdf([0.5, -0.2], 0, 2).sum())


def test_replay_forces_recorded_values_and_scores_them_under_new_arguments():
    """Test that a replay returns the recorded values and their log-probability under its own arguments."""
    recorder = DrawRecorder().prepare().run(1)
    recorder.draw('walker.kick', KICK, [0.0, 1.0], np.array([0.5]))
    replay = DrawRecorder(recorder.log_).prepare().run(1)
    forced = replay.draw('walker.kick', KICK, [0.0, 3.0], np.array([9.0]))
    assert forced.tolist() == [0.5]
    assert replay.log_probability_ == pytest.approx(stats.norm.logpdf(0.5, 0, 3))


@pytest.mark.parametrize(
    ('label', 'step', 'message'),
    [
        ('walker.jump', 1, r"draws 'walker\.jump', which the replayed trajectory lacks.*walker\.kick"),
        ('walker.kick', 2, 'At step 2'),
    ],
    ids=['name', 'step'],
)
def test_replay_rejects_draws_that_do_not_match(label, step, message):
    """Test that a replay raises a named error for a draw the log lacks or one at another step."""
    recorder = DrawRecorder().prepare().run(1)
    recorder.draw('walker.kick', KICK, [0.0, 1.0], np.array([0.5]))
    replay = DrawRecorder(recorder.log_).prepare().run(step)
    with pytest.raises(ValidationError, match=message):
        replay.draw(label, KICK, [0.0, 1.0], np.array([0.0]))


def test_recorder_scores_bernoulli_categorical_and_scipy_draws():
    """Test the log-probability of Bernoulli, categorical, and scipy.stats draws."""
    recorder = DrawRecorder().prepare()
    recorder.draw('a.flip', DrawSpec('bernoulli', 'flip'), [0.25], np.array([1.0, 0.0]))
    recorder.draw('a.pick', DrawSpec('categorical', 'pick'), [1.0, 3.0], np.array([1.0]))
    recorder.draw('a.wait', DrawSpec('draw', 'wait', 'expon', ('scale',)), [2.0], np.array([1.0]))
    expected = np.log(0.25) + np.log(0.75) + np.log(0.75) + stats.expon.logpdf(1.0, scale=2.0)
    assert recorder.log_probability_ == pytest.approx(expected)
    assert recorder.log_.draws['a.pick'].arguments['weights'].tolist() == [[1.0, 3.0]]
