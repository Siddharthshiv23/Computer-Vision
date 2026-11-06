import torch

from isp.learned.losses import gradient_loss, isp_loss, rgb_to_lab


def test_lab_known_values():
    rgb = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]).reshape(2, 3, 1, 1)
    lab = rgb_to_lab(rgb)
    assert torch.allclose(lab[0].flatten(), torch.tensor([100.0, 0.0, 0.0]), atol=0.1)
    assert torch.allclose(lab[1].flatten(), torch.tensor([53.24, 80.09, 67.20]), atol=0.2)


def test_identical_images_zero_loss():
    x = torch.rand(1, 3, 16, 16)
    assert isp_loss(x, x).item() < 1e-6


def test_loss_positive_and_finite_grad_with_black_pixels():
    target = torch.rand(1, 3, 16, 16)
    pred = torch.zeros(1, 3, 16, 16, requires_grad=True)
    loss = isp_loss(pred, target)
    loss.backward()
    assert loss.item() > 0
    assert torch.isfinite(pred.grad).all()


def test_colour_term_penalises_hue_shift_more_than_neutral_shift():
    target = torch.full((1, 3, 8, 8), 0.5)
    hue = target + torch.tensor([0.05, -0.05, 0.0]).view(1, 3, 1, 1)
    neutral = target + 0.05
    only_lab = dict(w_l1=0.0, w_lab=1.0, w_grad=0.0)
    assert isp_loss(hue, target, **only_lab) > isp_loss(neutral, target, **only_lab)


def test_gradient_loss_detects_blur():
    sharp = torch.zeros(1, 3, 8, 8)
    sharp[..., 4:] = 1.0
    blurred = sharp.clone()
    blurred[..., 3] = 0.5
    assert gradient_loss(blurred, sharp) > 0
    assert gradient_loss(sharp, sharp) == 0
