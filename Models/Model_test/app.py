import tkinter as tk
from tkinter import Label
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
from Models.config import ROOT_DIR

from .models import PoseModel, MediaPipeModel


class PoseApp:

    def __init__(self, root, model: PoseModel, transformer_model, class_names):
        self.root = root
        self.root.title("Pose Estimation App")
        self.model = model
        self.transformer_model = transformer_model
        self.class_names = class_names

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer_model.to(self.device)
        self.transformer_model.eval()

        self.keypoints_buffer = []

        self.status_label = Label(root, text="Action: No action detected", font=("Helvetica", 16))
        self.status_label.pack(pady=10)

        self.video_label = Label(root)
        self.video_label.pack()

        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            keypoints = self.model.predict(rgb_frame)
            for (x, y) in keypoints:
                cv2.circle(rgb_frame, (x, y), 5, (0, 255, 0), -1)

            self.keypoints_buffer.append(keypoints)
            if len(self.keypoints_buffer) > 35:
                self.keypoints_buffer.pop(0)

            if len(self.keypoints_buffer) == 35:
                self.process_buffer()

            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def process_buffer(self):
        reference_point = None
        for keypoints in self.keypoints_buffer:
            if keypoints and len(keypoints) > 0:
                reference_point = keypoints[0]
                break

        if not reference_point:
            reference_point = (0, 0)

        ref_x, ref_y = reference_point
        normalized_sequence = []

        for keypoints in self.keypoints_buffer:
            normalized_keypoints = []
            kp = keypoints[:21] if keypoints else []

            for x, y in kp:
                normalized_keypoints.append([x - ref_x, y - ref_y])

            while len(normalized_keypoints) < 21:
                normalized_keypoints.append([0.0, 0.0])

            normalized_sequence.append(normalized_keypoints)
        print('*' * 30)
        print(normalized_sequence)
        print('*' * 30 + '\n')
        tensor_data = torch.tensor(normalized_sequence, dtype=torch.float32)
        tensor_data = tensor_data.view(1, 35, 42).to(self.device)

        with torch.no_grad():
            outputs = self.transformer_model(tensor_data)
            probabilities = F.softmax(outputs, dim=1)
            max_prob, predicted_idx = torch.max(probabilities, 1)

            prob_value = max_prob.item()

            if prob_value >= 0.65:
                action = self.class_names[predicted_idx.item()]
                self.status_label.config(text=f"Action: {action} ({prob_value:.2f})", fg="green")
            else:
                self.status_label.config(text="Action: No action detected", fg="black")

    def on_closing(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    from Models.transformer_model import GestureTransformer

    root = tk.Tk()
    pose_model = MediaPipeModel()

    class_names = {'BIG_LEFT': 0, 'BIG_RIGHT': 1, 'CINEMA_MODE': 2, 'CLICK': 3, 'IDLE': 4, 'POINT_UP': 5,
                   'OPEN_PALM': 6, 'SITE_LEFT': 7, 'SMALL_LEFT': 8, 'SMALL_RIGHT': 9, 'SWIPE_LEFT': 10,
                   'SWIPE_RIGHT': 11, 'VIDEO': 12, 'VOLUME_DOWN': 13, 'VOLUME_UP': 14}

    idx_to_class = {v: k for k, v in class_names.items()}

    transformer = GestureTransformer(
        input_dim=42,
        d_model=128,
        nhead=8,
        num_layers=3,
        num_classes=len(idx_to_class)
    )

    try:
        transformer.load_state_dict(torch.load(ROOT_DIR / "best_gesture_model.pth", weights_only=True))
    except FileNotFoundError:
        pass

    app = PoseApp(root, pose_model, transformer, idx_to_class)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()