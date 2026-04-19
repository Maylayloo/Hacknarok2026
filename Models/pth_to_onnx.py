import torch
import onnx
from transformer_model import GestureTransformer
from config import ROOT_DIR
import os

os.chdir(ROOT_DIR.parent)

# 1. Inicjalizacja modelu
model = GestureTransformer(
    input_dim=42,
    d_model=128,
    nhead=8,
    num_layers=3,
    num_classes=2
)

# 2. Ładowanie wag
state_dict = torch.load(
    ROOT_DIR / "models/best_gesture_model.pth",
    map_location=torch.device('cpu'),
    weights_only=True
)
model.load_state_dict(state_dict)
model.eval()

# 3. Dummy input - upewnij się, że wymiary są identyczne z tymi w JS!
dummy_input = torch.randn(1, 35, 42)

# 4. EKSPORT (Zmienione parametry dla kompatybilności z JS)
onnx_path = str(ROOT_DIR / "models" / "gesture_model.onnx")

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,        # Zapisuje wagi wewnątrz pliku
    opset_version=17,          # ZMIANA: 17 jest stabilniejszy dla Web niż 18
    do_constant_folding=True,  # Optymalizuje graf
    input_names=['input_sequence'],
    output_names=['class_probabilities'],
    dynamic_axes=None          # ZMIANA: Wyłączamy dynamiczne osie dla stabilności
)

# 5. FINALNY SZLIF: Optymalizacja pliku pod kątem rozmiaru (usuwa metadane)
model_proto = onnx.load(onnx_path)
# Usuwamy informację o "pytorch", która mogła Cię zmylić w xxd (opcjonalne)
model_proto.producer_name = "onnx"
onnx.save(model_proto, onnx_path)

print(f"✅ Model wyeksportowany do: {onnx_path}")
onnx.checker.check_model(model_proto)
print("✅ Model jest prawidłowy!")