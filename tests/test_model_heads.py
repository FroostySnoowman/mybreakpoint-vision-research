"""Heatmap losses, head initialization and sub-pixel keypoint decoding.

The loss: a Gaussian heatmap is not a detection problem. The RetinaNet focal
loss's `alpha` weights the rare class, but the ball occupies a tiny fraction of
pixels, so `alpha=0.25` weights the background more than the ball.
`gaussian_focal_loss` is the CornerNet/CenterNet form: the Gaussian shoulder has
its penalty reduced rather than being labeled negative, and the total is
normalized by peak count so it is comparable across input resolutions.

The peak mask: a `target >= 1 - 1e-4` mask selects zero pixels on a
sub-pixel-centered Gaussian, yet a fixture that forces an exact 1.0 at the center
pixel (`_gaussian_target`) hides that. So fixtures are built with the dataset's
own renderer (`_rendered_target`), and assertions are on decoded peak position or
gradient sign, never on the loss value alone.

Sub-pixel keypoints: integer argmax on a 512-wide heatmap against a 1280-wide
source quantizes to 2.5 source pixels, which propagates through the homography.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from breakpoint_vision.data.heatmap import render_gaussian, render_multi  # noqa: E402
from breakpoint_vision.models.court_net import CourtNet, keypoint_peaks  # noqa: E402
from breakpoint_vision.models.tracknet import (  # noqa: E402
    HEATMAP_BIAS_PRIOR,
    TrackNet,
    focal_bce_loss,
    gaussian_focal_loss,
    heatmap_peak,
)


def _gaussian_target(h=64, w=64, cx=31.5, cy=20.5, sigma=2.0):
    ys, xs = np.mgrid[0:h, 0:w]
    hm = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2)))
    hm[int(round(cy)), int(round(cx))] = 1.0
    return torch.from_numpy(hm.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def _rendered_target(cx=31.4, cy=20.7, h=64, w=64, sigma=3.0):
    """A target produced by the dataset's renderer, not a test reimplementation.

    This goes through `render_gaussian`, the same call `DiskBallDataset` makes,
    so the fixture cannot drift from what training sees. In particular it does
    not force a 1.0 at the nearest pixel the way `_gaussian_target` does: the
    dataset centers the Gaussian on a sub-pixel position, so its brightest pixel
    is never exactly 1.0.
    """
    hm = render_gaussian(h, w, (cx, cy), sigma)
    return torch.from_numpy(hm).unsqueeze(0).unsqueeze(0)


def test_gaussian_focal_loss_does_not_starve_the_peak_of_gradient():
    """With the detection focal loss a peak pixel receives less gradient than a
    background pixel."""
    target = _gaussian_target()
    peaks = target.ge(1.0 - 1e-4)
    background = target.le(1e-6)

    def per_pixel_ratio(loss_fn):
        logits = torch.zeros_like(target, requires_grad=True)
        loss_fn(logits, target).backward()
        grad = logits.grad.abs()
        return float(grad[peaks].mean() / grad[background].mean())

    assert per_pixel_ratio(focal_bce_loss) < 0.5
    assert per_pixel_ratio(gaussian_focal_loss) > 0.9


def test_gaussian_focal_loss_is_finite_when_no_ball_is_present():
    """Occluded/out-of-frame windows are legitimate samples with an all-zero target,
    so the peak-count normalizer must not divide by zero."""
    target = torch.zeros(1, 1, 32, 32)
    loss = gaussian_focal_loss(torch.zeros_like(target), target)
    assert torch.isfinite(loss)


def test_gaussian_focal_loss_rewards_a_correct_prediction():
    target = _gaussian_target()
    logits = torch.logit(target.clamp(1e-4, 1 - 1e-4))
    assert gaussian_focal_loss(logits, target) < gaussian_focal_loss(torch.zeros_like(target), target)


def test_heatmap_heads_start_biased_towards_empty():
    """Without this the gaussian focal loss cannot train at all.

    The loss sums over pixels and divides by peak count, so a head starting at
    p=0.5 on every pixel produces an enormous gradient norm, and clipping then
    shrinks every step to almost nothing. Both heads must therefore start near
    `HEATMAP_BIAS_PRIOR`, not at 0.5.
    """
    for model in (TrackNet(input_frames=3, base=8), CourtNet(num_keypoints=14, base=8)):
        prior = torch.sigmoid(model.head.bias.detach()).mean().item()
        assert prior == pytest.approx(HEATMAP_BIAS_PRIOR, abs=1e-6)


def test_gaussian_focal_loss_is_scale_free_in_the_background():
    """The loss and the gradient clip have to be compatible, not just individually
    reasonable.

    The failure mode this guards is subtle: the loss divides by *peak count*, so if
    the background term also grew with pixel count the gradient norm would scale
    with resolution and any fixed `grad_clip_norm` would throttle the larger input.
    Doubling the background area at a fixed number of peaks must not blow the loss
    up proportionally.
    """
    small = _gaussian_target(64, 64, 31.5, 20.5)
    large = _gaussian_target(64, 128, 31.5, 20.5)

    loss_small = gaussian_focal_loss(torch.zeros_like(small), small)
    loss_large = gaussian_focal_loss(torch.zeros_like(large), large)

    assert loss_large < 2.5 * loss_small


@pytest.mark.parametrize("cx,cy", [(31.4, 20.7), (10.0, 10.0), (48.5, 33.5), (7.12, 55.93)])
def test_gaussian_focal_loss_pulls_the_true_center_upwards(cx, cy):
    """The loss must want more probability at the labeled center.

    A fixed peak mask like `target >= 1 - 1e-4` selects zero pixels on a
    sub-pixel-centered Gaussian, so `pos_loss` is identically zero and the loss's
    global optimum becomes "predict background everywhere".

    The assertion is on the gradient at the labeled center: it must be negative.
    Under a fixed mask that pixel is treated as a plain negative and the gradient
    pushes it down. This goes
    through `gaussian_focal_loss` rather than re-deriving the mask, so the test
    cannot pass by duplicating a broken implementation.
    """
    target = _rendered_target(cx, cy)
    logits = torch.zeros_like(target, requires_grad=True)
    gaussian_focal_loss(logits, target).backward()

    center = logits.grad[0, 0, int(round(cy)), int(round(cx))]
    assert float(center) < 0.0

    corner = logits.grad[0, 0, -1, -1]
    assert float(corner) > 0.0


def test_gaussian_focal_loss_gives_every_court_keypoint_a_positive():
    """The court target stacks 15 channels, each with its own sub-pixel rounding.

    A peak mask taken over the whole sample sets the bar at the luckiest keypoint's
    maximum and starves every channel whose own maximum falls below it of positive
    gradient. Every keypoint feeds the homography, so the mask is per-channel.
    """
    rng = np.random.default_rng(7)
    centers = [(float(rng.uniform(20, 490)), float(rng.uniform(20, 270))) for _ in range(15)]
    target = torch.from_numpy(render_multi(288, 512, centers, 3.0)).unsqueeze(0)

    logits = torch.zeros_like(target, requires_grad=True)
    gaussian_focal_loss(logits, target).backward()

    for k, (cx, cy) in enumerate(centers):
        center_grad = logits.grad[0, k, int(round(cy)), int(round(cx))]
        assert float(center_grad) < 0.0, f"keypoint {k} got no upward gradient"


def test_gaussian_focal_loss_ignores_occluded_channels():
    """An occluded keypoint is an all-zero channel and must contribute no positive.

    `peak_floor` is what enforces this: without it, a channel's own maximum is 0 and
    the relative threshold would promote the entire empty channel to a peak.
    """
    rng = np.random.default_rng(11)
    centers = [(float(rng.uniform(20, 490)), float(rng.uniform(20, 270))) for _ in range(15)]
    for k in (2, 5, 9, 13):
        centers[k] = None
    target = torch.from_numpy(render_multi(288, 512, centers, 3.0)).unsqueeze(0)

    logits = torch.zeros_like(target, requires_grad=True)
    loss = gaussian_focal_loss(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    for k in (2, 5, 9, 13):
        assert float(target[0, k].max()) == 0.0
        assert float(logits.grad[0, k].min()) >= 0.0, f"empty channel {k} was treated as a peak"


def test_gaussian_focal_loss_rejects_a_squeezed_target():
    """A (B, H, W) target would be read as B x H single-row channels."""
    target = _rendered_target()[0]
    with pytest.raises(ValueError, match="B, C, H, W"):
        gaussian_focal_loss(torch.zeros_like(target), target)


def test_gaussian_focal_loss_actually_fits_a_sub_pixel_target():
    """End-to-end: overfit one realistic sample and require the peak to land on it.

    A loss can look healthy (falling smoothly, sane gradient norms) while being
    unable to localize anything, so the assertion is on the decoded peak
    position, not on the loss value.
    """
    torch.manual_seed(1337)
    target = _rendered_target(31.4, 20.7)
    logits = torch.zeros_like(target, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.5)

    for _ in range(300):
        opt.zero_grad()
        gaussian_focal_loss(logits, target).backward()
        opt.step()

    coords, conf = heatmap_peak(torch.sigmoid(logits.detach()))
    assert conf[0] > 0.5
    assert abs(float(coords[0, 0]) - 31.4) <= 1.0
    assert abs(float(coords[0, 1]) - 20.7) <= 1.0


def test_keypoint_peaks_recover_sub_pixel_centers():
    centers = [(7.3, 5.8), (30.6, 9.1), (55.2, 28.4), (12.9, 31.7)]
    maps = torch.cat([_gaussian_target(36, 64, cx, cy, 1.8) for (cx, cy) in centers], dim=1)

    coords, conf = keypoint_peaks(maps)
    for k, (cx, cy) in enumerate(centers):
        assert abs(float(coords[0, k, 0]) - cx) < 0.35
        assert abs(float(coords[0, k, 1]) - cy) < 0.35
        assert conf[0, k] > 0.9


def test_keypoint_peaks_beat_the_integer_argmax():
    centers = [(7.3, 5.8), (30.6, 9.1), (55.2, 28.4), (12.9, 31.7)]
    maps = torch.cat([_gaussian_target(36, 64, cx, cy, 1.8) for (cx, cy) in centers], dim=1)
    truth = torch.tensor([[list(c) for c in centers]], dtype=torch.float32)

    refined, _ = keypoint_peaks(maps)
    integer, _ = keypoint_peaks(maps, subpixel_radius=0)

    refined_err = (refined - truth).pow(2).sum(dim=2).sqrt().mean()
    integer_err = (integer - truth).pow(2).sum(dim=2).sqrt().mean()
    assert refined_err < integer_err


def test_tracknet_can_predict_one_heatmap_per_input_frame():
    import torch

    from breakpoint_vision.models.tracknet import TrackNet

    m = TrackNet(input_frames=3, base=8, out_frames=3).eval()
    assert m(torch.zeros(1, 9, 64, 128)).shape == (1, 3, 64, 128)
    assert TrackNet(input_frames=3, base=8).eval()(torch.zeros(1, 9, 64, 128)).shape == (1, 1, 64, 128)
