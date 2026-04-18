import tkinter as tk
from tkinter import Label
import cv2
import numpy as np
from PIL import Image, ImageTk
from .models import PoseModel, MediaPipeModel


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
            print('*' * 20)
            print(keypoints)
            print('*' * 20)
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