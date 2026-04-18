import sys
import urllib.request
from pathlib import Path
from Models.config import ROOT_DIR

def download_mediapipe_weights():

    filename = "pose_landmarker_lite.task"

    if (ROOT_DIR / filename).exists():
        print("Model weights already downloaded...")
        return

    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    print("Downloading model...")
    urllib.request.urlretrieve(url, filename)
    print("Download complete!")