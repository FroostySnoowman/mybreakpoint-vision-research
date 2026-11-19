"""The numpy-only synthetic data path: synthetic clip -> trajectory.

Runs without any torch dependency.
"""

from breakpoint_vision.data.synthetic import clip_to_trajectory, make_clip


def test_synthetic_clip_shapes():
    clip = make_clip(length=8, height=120, width=160, seed=3)
    assert clip.frames.shape == (8, 120, 160, 3)
    assert len(clip.positions) == 8


def test_trajectory_conversion_marks_invisible():
    clip = make_clip(length=10, height=120, width=160, seed=1, invisible_prob=1.0)
    traj = clip_to_trajectory(clip)
    assert all(not t[3] for t in traj)


def test_visible_clip_produces_trajectory():
    clip = make_clip(length=10, height=200, width=320, seed=5, invisible_prob=0.0)
    traj = clip_to_trajectory(clip)
    visible = [t for t in traj if t[3]]
    assert len(visible) >= 8
