import pytest

from breakpoint_vision import params
from breakpoint_vision.ball_track import BallTrackerConfig
from breakpoint_vision.points.decode import DecodeConfig


def test_missing_tuned_file_raises_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(params, "TUNED_PATH", str(tmp_path / "tuned.yaml"))
    with pytest.raises(FileNotFoundError, match="reproducibility.md"):
        params.tuned("decode")
    with pytest.raises(FileNotFoundError):
        DecodeConfig.tuned()


def test_tuned_sections_build_configs(tmp_path, monkeypatch):
    path = tmp_path / "tuned.yaml"
    path.write_text(
        "ball_tracker:\n"
        "  min_candidate_confidence: 0.03\n"
        "  seed_confidence: 0.6\n"
        "  keep_confidence: 0.1\n"
        "  gate_fraction: 0.03\n"
        "  gate_growth_fraction: 0.015\n"
        "  coast_frames: 3\n"
        "  confirm_detections: 3\n"
    )
    monkeypatch.setattr(params, "TUNED_PATH", str(path))
    cfg = BallTrackerConfig.tuned(max_candidates=2)
    assert cfg.seed_confidence == 0.6 and cfg.max_candidates == 2
    with pytest.raises(KeyError):
        params.tuned("decode")
