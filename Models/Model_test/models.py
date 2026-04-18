import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerOptions
import abc
import numpy as np
from .download_mediapipe import download_mediapipe_weights



class PoseModel(abc.ABC):

    @abc.abstractmethod
    def predict(self, image: np.ndarray) -> list:
        pass




class MediaPipeModel(PoseModel):

    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
        hand_options = HandLandmarkerOptions(base_options, running_mode=vision.RunningMode.IMAGE, num_hands=1,
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
