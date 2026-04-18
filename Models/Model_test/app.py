import abc
import tkinter as tk
from tkinter import Label
import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerOptions

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

    def __init__(self, root, model: PoseModel):
        self.root = root
        self.root.title("Pose Estimation App")
        self.model = model

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

            img = Image.fromarray(rgb_frame)
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