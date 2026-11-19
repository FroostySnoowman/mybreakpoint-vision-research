import numpy as np

from breakpoint_vision.data.lighting import LightingConfig, LightingDegrader, LightingSample, linear_to_srgb, srgb_to_linear

ILLUSTRATIVE = dict(
    night_exposure=(0.05, 0.4),
    harsh_exposure=(1.5, 2.5),
    read_noise=(0.001, 0.003),
    dark_read_noise=(0.002, 0.006),
    photons_at_white=(3000.0, 15000.0),
    residual_exposure=(0.65, 1.0),
    capture_area_ratio=(4.0, 12.0),
    denoise_sigma=(0.4, 0.7),
)


def _cfg(**overrides) -> LightingConfig:
    return LightingConfig(**{**ILLUSTRATIVE, **overrides})


def _frame(seed=0, h=72, w=128):
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), 120, np.uint8)
    base[h // 2:, :, :] = (40, 90, 160)
    base += rng.integers(0, 20, size=base.shape, dtype=np.uint8)
    return base


def _sample(**over):
    s = LightingSample(exposure=1.0, gain=np.ones(3, np.float32), read_noise=0.0, photons=1e7, shadow=None, vignette=0.0, denoise_sigma=0.0, jpeg_quality=None, auto_gain_cap=12.0)
    for k, v in over.items():
        setattr(s, k, v)
    return s


def test_srgb_roundtrip_is_exact_enough():
    x = np.linspace(0, 1, 1001)
    assert np.max(np.abs(linear_to_srgb(srgb_to_linear(x)) - x)) < 1e-6


def test_output_shape_and_dtype_are_preserved():
    d = LightingDegrader(_cfg(), seed=1)
    out = d.apply_window([_frame(), _frame(1)])
    assert len(out) == 2
    assert all(o.shape == (72, 128, 3) and o.dtype == np.uint8 for o in out)


def test_same_seed_gives_same_result():
    a = LightingDegrader(_cfg(), seed=7).apply_window([_frame()])[0]
    b = LightingDegrader(_cfg(), seed=7).apply_window([_frame()])[0]
    assert np.array_equal(a, b)


def test_identity_sample_keeps_the_frame():
    f = _frame()
    out = LightingDegrader(_cfg(), seed=0).apply_frame(f, _sample())
    assert np.abs(out.astype(int) - f.astype(int)).max() <= 2


def test_dark_exposure_without_camera_gain_darkens_the_frame():
    f = _frame()
    out = LightingDegrader(_cfg(), seed=0).apply_frame(f, _sample(exposure=0.05, auto_gain_cap=1.0))
    assert out.mean() < f.mean() * 0.5


def test_dark_exposure_with_full_camera_gain_restores_brightness_but_adds_noise():
    f = _frame()
    out = LightingDegrader(_cfg(), seed=0).apply_frame(f, _sample(exposure=0.05, photons=400.0, read_noise=0.01))
    assert abs(out.mean() - f.mean()) < 40
    assert out.astype(float).std() > f.astype(float).std() * 1.3


def test_harsh_exposure_clips_highlights():
    f = np.full((32, 32, 3), 200, np.uint8)
    out = LightingDegrader(_cfg(), seed=0).apply_frame(f, _sample(exposure=2.5))
    assert (out == 255).mean() > 0.9


def test_global_parameters_are_shared_across_a_window():
    cfg = _cfg(night_prob=1.0, harsh_prob=0.0, read_noise=(0.0, 0.0), dark_read_noise=(0.0, 0.0), photons_at_white=(1e7, 1e7), jpeg_prob=0.0, denoise_prob=0.0)
    f = _frame()
    for seed in range(5):
        a, b = LightingDegrader(cfg, seed=seed).apply_window([f, f])
        assert np.abs(a.astype(int) - b.astype(int)).mean() < 1.0


def test_warm_cast_shifts_red_above_blue():
    f = np.full((32, 32, 3), 128, np.uint8)
    out = LightingDegrader(_cfg(), seed=0).apply_frame(f, _sample(gain=np.array([1.25, 1.0, 0.7], np.float32)))
    assert out[..., 0].mean() > out[..., 2].mean() + 20


def test_shadow_mask_darkens_part_of_the_frame_only():
    d = LightingDegrader(_cfg(shadow_prob=1.0), seed=3)
    mask = d._shadow(72, 128)
    assert mask.min() < 0.7 and mask.max() > 0.95


def test_sampled_night_keeps_realistic_brightness_after_auto_exposure():
    f = _frame()
    for seed in range(20):
        out = LightingDegrader(_cfg(night_prob=1.0, harsh_prob=0.0, shadow_prob=0.0, vignette_prob=0.0, cast_prob=0.0), seed=seed).apply_window([f])[0]
        assert out.mean() > f.mean() * 0.6


def test_capture_resolution_averages_away_most_sensor_noise():
    f = np.full((64, 64, 3), 128, np.uint8)
    fine = _sample(exposure=0.1, auto_gain_cap=10.0, photons=1000.0 * 10.0, read_noise=0.01 / np.sqrt(10.0))
    coarse = _sample(exposure=0.1, auto_gain_cap=10.0, photons=1000.0, read_noise=0.01)
    a = LightingDegrader(_cfg(), seed=0).apply_frame(f, fine).astype(float).std()
    b = LightingDegrader(_cfg(), seed=0).apply_frame(f, coarse).astype(float).std()
    assert a < b * 0.5
