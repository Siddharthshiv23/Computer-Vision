import numpy as np
import pytest
import torch

from isp.models.liteisp import STUDENT_WIDTH, LiteISPNet
from isp.models.rawnet import RawNet

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")


@pytest.mark.parametrize("net,shape", [(LiteISPNet(STUDENT_WIDTH), (1, 4, 40, 56)), (RawNet(), (1, 5, 20, 28))])
def test_onnx_opset17_matches_torch_and_is_dynamic(net, shape, tmp_path):
    net.eval()
    x = torch.rand(*shape)
    path = str(tmp_path / "m.onnx")
    torch.onnx.export(net, x, path, opset_version=17, input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {2: "h", 3: "w"}, "y": {2: "h", 3: "w"}}, dynamo=False)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"x": x.numpy()})[0]
    with torch.no_grad():
        want = net(x).numpy()
    assert np.abs(got - want).max() < 1e-4
    other = torch.rand(shape[0], shape[1], shape[2] + 8, shape[3] + 4)
    with torch.no_grad():
        want2 = net(other).numpy()
    assert np.abs(sess.run(None, {"x": other.numpy()})[0] - want2).max() < 1e-4
