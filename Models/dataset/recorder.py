import abc
import tkinter as tk
from tkinter import Label, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerOptions

import time
import os
import json
import datetime
import uuid
import re

from .download_mediapipe import download_mediapipe_weights


class PoseModel(abc.ABC):

    @abc.abstractmethod
    def predict(self, image: np.ndarray) -> list:
        pass


class MediaPipeModel(PoseModel):

    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
        hand_options = HandLandmarkerOptions(base_options, running_mode=vision.RunningMode.IMAGE, num_hands=2,
                                             min_hand_detection_confidence=0.5,
                                             min_hand_presence_confidence=0.5,
                                             min_tracking_confidence=0.5
                                             )
        download_mediapipe_weights()
        self.detector = vision.HandLandmarker.create_from_options(hand_options)

    def predict(self, image: np.ndarray) -> list:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        detection_result = self.detector.detect(mp_image)

        keypoints = []

        if detection_result.hand_landmarks:
            h, w, _ = image.shape

            for hand in detection_result.hand_landmarks:
                for landmark in hand:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    keypoints.append((cx, cy))

        return keypoints


class PoseApp:
    AVAILABLE_LABELS = [
        "POINT_UP",
        "SWIPE_RIGHT",
        "SWIPE_LEFT",
        "SWIPE_UP",
        "SWIPE_DOWN",
        "FIST",
        "OPEN_PALM"
    ]

    def __init__(self, root, model: PoseModel):
        self.root = root
        self.root.title("Pose Estimation App - Dataset Creator")
        self.model = model

        self.control_frame = tk.Frame(root)
        self.control_frame.pack(pady=5)

        tk.Label(self.control_frame, text="Label:").pack(side=tk.LEFT)

        self.label_var = tk.StringVar()

        self.label_dropdown = ttk.Combobox(
            self.control_frame,
            textvariable=self.label_var,
            values=self.AVAILABLE_LABELS,
            state="readonly",
            width=15
        )
        self.label_dropdown.pack(side=tk.LEFT, padx=5)

        if self.AVAILABLE_LABELS:
            self.label_dropdown.current(0)
        # ----------------------------------

        self.record_btn = tk.Button(self.control_frame, text="Record (35 frames)", command=self.start_recording)
        self.record_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self.control_frame, text="Status: Oczekiwanie", fg="black")
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.video_label = Label(root)
        self.video_label.pack()

        self.cap = cv2.VideoCapture(0)

        self.is_recording = False
        self.session_dir = ""
        self.current_label = ""
        self.raw_frames_buffer = []

        self.update_frame()

    def start_recording(self):
        if self.is_recording:
            return

        self.current_label = self.label_var.get().strip()
        if not self.current_label:
            self.status_label.config(text="Wybierz etykietę!", fg="red")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]
        self.session_dir = os.path.join("dataset", f"{self.current_label}_{timestamp}_{unique_id}")
        os.makedirs(self.session_dir, exist_ok=True)

        self.raw_frames_buffer = []
        self.is_recording = True

        self.status_label.config(text="Nagrywanie (37 klatek)...", fg="red")
        self.record_btn.config(state=tk.DISABLED)
        self.label_dropdown.config(state=tk.DISABLED)

    def stop_recording(self):
        self.is_recording = False
        self.status_label.config(text="Przetwarzanie i zapis...", fg="blue")

        frames_to_process = self.raw_frames_buffer[2: 37]

        reference_point = None
        for frame, keypoints in frames_to_process:
            if keypoints and len(keypoints) > 0:
                reference_point = keypoints[0]
                break

        if not reference_point:
            reference_point = (0, 0)
            print("Ostrzeżenie: Nie wykryto dłoni na żadnej klatce!")

        ref_x, ref_y = reference_point
        recording_data = []

        for i, (frame, keypoints) in enumerate(frames_to_process):
            filename = f"frame_{i:04d}.jpg"
            filepath = os.path.join(self.session_dir, filename)
            cv2.imwrite(filepath, frame)

            normalized_keypoints = []
            for x, y in keypoints:
                normalized_keypoints.append([x - ref_x, y - ref_y])

            recording_data.append({
                "photo_path": filepath,
                "key_points": normalized_keypoints
            })

        json_data = {
            "klatki": recording_data,
            "label": self.current_label
        }

        json_str = json.dumps(json_data, indent=4)
        json_str = re.sub(r'\[\s+(-?\d+),\s+(-?\d+)\s+\]', r'[\1, \2]', json_str)

        json_path = os.path.join(self.session_dir, "data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_str)

        self.status_label.config(text=f"Zapisano 35 klatek!", fg="green")
        self.record_btn.config(state=tk.NORMAL)
        self.label_dropdown.config(state="readonly")  # Odblokowujemy listę po nagraniu
        print(f"Zakończono. Pliki zapisano w: {self.session_dir}")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            keypoints = self.model.predict(rgb_frame)

            if self.is_recording:
                self.raw_frames_buffer.append((frame.copy(), keypoints))

                if len(self.raw_frames_buffer) >= 37:
                    self.stop_recording()

            display_frame = rgb_frame.copy()
            for (x, y) in keypoints:
                cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)

            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def on_closing(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    pose_model = MediaPipeModel()
    app = PoseApp(root, pose_model)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()