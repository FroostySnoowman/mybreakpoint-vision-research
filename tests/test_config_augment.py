"""The `augment:` block in a training YAML actually reaches the training loop.

A config block that is silently ignored is the worst kind of bug here: setting
`enabled: false` would appear to disable augmentation and change nothing, and the
run would be attributed to the wrong settings. These tests load the configs in
`configs/`, so a future restructuring of them fails here rather than in a long run.
"""

import os

from breakpoint_vision.config import load_ball_train_config, load_court_train_config

CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def test_the_ball_config_enables_augmentation():
    cfg = load_ball_train_config(os.path.join(CONFIGS, "ball_train.yaml"))
    assert cfg.augment.enabled
    assert cfg.augment.horizontal_flip_prob > 0


def test_the_court_config_enables_augmentation():
    cfg = load_court_train_config(os.path.join(CONFIGS, "court_train.yaml"))
    assert cfg.augment.enabled
    assert cfg.augment.horizontal_flip_prob > 0


def test_augment_is_read_from_the_document_root(tmp_path):
    """Where the configs in `configs/` put it, beside `ball_train:` rather than inside."""
    path = tmp_path / "root.yaml"
    path.write_text("ball_train:\n  epochs: 3\naugment:\n  enabled: false\n  rotation_degrees: 99.0\n")
    cfg = load_ball_train_config(str(path))
    assert cfg.epochs == 3
    assert not cfg.augment.enabled
    assert cfg.augment.rotation_degrees == 99.0


def test_augment_is_read_from_inside_the_training_section(tmp_path):
    path = tmp_path / "nested.yaml"
    path.write_text("court_train:\n  epochs: 2\n  augment:\n    enabled: false\n")
    cfg = load_court_train_config(str(path))
    assert cfg.epochs == 2
    assert not cfg.augment.enabled


def test_a_nested_block_wins_over_a_root_block(tmp_path):
    """The more specific location is the one a reader would expect to apply."""
    path = tmp_path / "both.yaml"
    path.write_text(
        "augment:\n  rotation_degrees: 1.0\n"
        "ball_train:\n  epochs: 1\n  augment:\n    rotation_degrees: 2.0\n"
    )
    assert load_ball_train_config(str(path)).augment.rotation_degrees == 2.0


def test_omitting_the_block_leaves_augmentation_on_by_default(tmp_path):
    path = tmp_path / "bare.yaml"
    path.write_text("ball_train:\n  epochs: 1\n")
    assert load_ball_train_config(str(path)).augment.enabled


def test_an_unknown_augment_key_does_not_crash_a_run(tmp_path):
    """A typo should not take down a training job hours in."""
    path = tmp_path / "typo.yaml"
    path.write_text("ball_train:\n  epochs: 1\naugment:\n  rotaton_degrees: 12.0\n")
    cfg = load_ball_train_config(str(path))
    assert cfg.augment.rotation_degrees == 4.0
