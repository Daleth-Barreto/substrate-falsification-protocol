import os
import sys
import torch

jit_path = sys.argv[1] if len(sys.argv) > 1 else "../third_party/g1_deploy_mujoco/checkpoint/policy.pt"
onnx_path = sys.argv[2] if len(sys.argv) > 2 else "exported/policy.onnx"

os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

ts = torch.jit.load(jit_path).eval()
dummy = torch.randn(1, 480)

traced = torch.jit.trace(ts, dummy)
with torch.inference_mode():
    torch.onnx.export(
        traced,
        dummy,
        onnx_path,
        input_names=["obs"],
        output_names=["actions"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
print("OK exported:", onnx_path)