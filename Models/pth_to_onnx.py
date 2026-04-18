import torch
import onnx

# Załaduj model PyTorch
model = torch.load('./Models/best_gesture_model.pth')  # lub torch.load('gesture_model.pth')
model.eval()

# Utwórz dummy input (dostosuj do rozmiaru wejścia Twojego modelu)
dummy_input = torch.randn(1, 30, 42)  # Przykład: obraz 224x224x3

# Eksportuj do ONNX
torch.onnx.export(
    model,
    dummy_input,
    'models/gesture_model_v2.onnx',
    export_params=True,
    opset_version=13,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# Sprawdź czy model jest prawidłowy
onnx_model = onnx.load('models/gesture_model_v2.onnx')
onnx.checker.check_model(onnx_model)
print("✅ Model ONNX jest prawidłowy!")
