import torch
from Models.transformer_model import GestureTransformer

model = GestureTransformer(input_dim=42, d_model=128, nhead=8, num_layers=3, num_classes=2)

model.load_state_dict(torch.load("best_gesture_model.pth", map_location=torch.device('cpu'), weights_only=True))
model.eval()

dummy_input = torch.randn(1, 30, 42)

torch.onnx.export(
    model,
    dummy_input,
    "gesture_model.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input_sequence'],
    output_names=['class_probabilities']
)