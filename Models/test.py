import onnxruntime as ort
import numpy as np
import torch

dummy_input_np = np.random.randn(1, 35, 42).astype(np.float32)
dummy_input_pt = torch.from_numpy(dummy_input_np)

from Models.transformer_model import GestureTransformer
model_pt = GestureTransformer(input_dim=42, d_model=128, nhead=8, num_layers=3, num_classes=15)
model_pt.load_state_dict(torch.load("best_gesture_model.pth", map_location='cpu', weights_only=True))
model_pt.eval()

with torch.no_grad():
    pt_output = model_pt(dummy_input_pt)
    print("PyTorch Output:")
    print(pt_output)

session = ort.InferenceSession("gesture_model.onnx")
ort_inputs = {session.get_inputs()[0].name: dummy_input_np}
ort_output = session.run(None, ort_inputs)

print("ONNX Output:")
print(ort_output[0])

import onnxruntime as ort
import numpy as np

dummy_input_np = np.random.randn(1, 35, 42).astype(np.float32)

session = ort.InferenceSession("gesture_model.with_runtime_opt.ort")
ort_inputs = {session.get_inputs()[0].name: dummy_input_np}
ort_output = session.run(None, ort_inputs)

print("ORT Output:")
print(ort_output[0])