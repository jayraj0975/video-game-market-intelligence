import numpy as np

from features import FEATURES, POST_RELEASE
from config import LEAKY_COLUMNS
from train import capture_at_budget


def test_perfect_ranking_captures_every_hit_it_can():
    y = np.array([1] * 10 + [0] * 90)
    scores = np.linspace(1, 0, 100)
    cap = capture_at_budget(y, scores, 0.10)
    assert cap["hits_caught"] == 10 and cap["recall"] == 1.0
    assert cap["lift"] == 10.0


def test_constant_scores_are_no_better_than_random():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.12).astype(int)
    cap = capture_at_budget(y, np.zeros(5000), 0.10)
    assert abs(cap["lift"] - 1.0) < 0.25


def test_feature_set_excludes_target_and_post_release_columns():
    assert not set(FEATURES) & set(LEAKY_COLUMNS)
    assert not set(FEATURES) & set(POST_RELEASE)
