import numpy as np

from breakpoint_vision.data.heatmap import render_gaussian, render_multi


def test_gaussian_peak_at_center():
    hm = render_gaussian(64, 64, (32, 20), sigma=3.0)
    peak = np.unravel_index(np.argmax(hm), hm.shape)
    assert peak == (20, 32)
    assert abs(hm.max() - 1.0) < 1e-5


def test_none_center_is_all_zero():
    hm = render_gaussian(32, 32, None, sigma=3.0)
    assert hm.max() == 0.0


def test_out_of_bounds_center_is_zero():
    hm = render_gaussian(32, 32, (100, 100), sigma=3.0)
    assert hm.max() == 0.0


def test_render_multi_channels():
    stack = render_multi(32, 32, [(5, 5), None, (20, 20)], sigma=2.0)
    assert stack.shape == (3, 32, 32)
    assert stack[1].max() == 0.0
    assert stack[0].max() > 0.5
