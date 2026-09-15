import numpy as np
from pathlib import Path


class ANNPolicy:
    def __init__(self, path):
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".onnx":
            self._load_onnx(path)
        elif suffix == ".pt":
            self._load_torch(path)
        else:
            self._load_onnx(path)

    def _load_onnx(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.backend = "onnx"
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def _load_torch(self, path):
        import torch
        self.model = torch.jit.load(str(path), map_location="cpu").eval()
        self.backend = "torch"

    def act(self, stacked_obs):
        x = np.asarray(stacked_obs, dtype=np.float32).reshape(1, -1)
        if self.backend == "onnx":
            out = self.sess.run([self.output_name], {self.input_name: x})[0]
            return np.asarray(out, dtype=float).ravel()
        import torch
        with torch.inference_mode():
            out = self.model(torch.from_numpy(x))
        return out.numpy().ravel()