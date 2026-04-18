import abc
import tkinter as tk
from tkinter import Label
import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp


class PoseModel(abc.ABC):

    @abc.abstractmethod
    def predict(self, image: np.ndarray) -> list:
        pass


class MediaPipeModel(PoseModel):

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def predict(self, image: np.ndarray) -> list:
        results = self.pose.process(image)
        keypoints = []
        if results.pose_landmarks:
            h, w, _ = image.shape
            for landmark in results.pose_landmarks.landmark:
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