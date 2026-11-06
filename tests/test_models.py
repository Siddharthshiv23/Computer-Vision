import pytest
import torch

from isp.models.liteisp import STUDENT_WIDTH, TEACHER_WIDTH, LiteISPNet
from isp.models.rawnet import RawNet


def count(m):
    return sum(p.numel() for p in m.parameters())


def test_student_under_200k_params():
    n = count(LiteISPNet(STUDENT_WIDTH))
    assert 0 < n < 200_000


def test_teacher_larger_than_student():
    assert count(LiteISPNet(TEACHER_WIDTH)) > 2 * count(LiteISPNet(STUDENT_WIDTH))


@pytest.mark.parametrize("hw", [(32, 32), (33, 47)])
def test_liteisp_shape_any_size(hw):
    out = LiteISPNet(STUDENT_WIDTH)(torch.rand(2, 4, *hw))
    assert out.shape == (2, 3, *hw)


def test_liteisp_starts_near_identity():
    net = LiteISPNet(STUDENT_WIDTH).eval()
    x = torch.rand(1, 4, 32, 32)
    assert torch.allclose(net(x), x[:, :3], atol=0.1)


def test_colour_head_bias_shifts_colour_globally():
    net = LiteISPNet(STUDENT_WIDTH).eval()
    with torch.no_grad():
        net.colour_head[-1].bias[9] += 0.1  # red bias
    x = torch.rand(1, 4, 32, 32)
    assert (net(x)[:, 0] - x[:, 0]).mean() > 0.05


def test_rawnet_shape_and_params():
    net = RawNet()
    out = net(torch.rand(2, 5, 16, 24))
    assert out.shape == (2, 3, 32, 48)
    assert count(net) > 0


def test_gradients_flow():
    net = LiteISPNet(STUDENT_WIDTH)
    net(torch.rand(1, 4, 16, 16)).mean().backward()
    assert all(p.grad is not None for p in net.parameters())
