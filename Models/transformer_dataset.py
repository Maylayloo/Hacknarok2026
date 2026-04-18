import os
import json
import torch
from torch.utils.data import Dataset


class JSONGestureDataset(Dataset):
    def __init__(self, root_dir, target_frames=35):
        self.root_dir = root_dir
        self.target_frames = target_frames
        self.data_paths = []
        self.labels = []
        self.class_to_idx = {}

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                json_path = os.path.join(folder_path, "data.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        label_str = data.get("label")

                        if label_str not in self.class_to_idx:
                            self.class_to_idx[label_str] = len(self.class_to_idx)

                        self.data_paths.append(json_path)
                        self.labels.append(self.class_to_idx[label_str])

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        json_path = self.data_paths[idx]
        label_idx = self.labels[idx]

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        frames = data.get("klatki", [])
        processed_frames = []

        for frame in frames:
            kp = frame.get("key_points", [])

            if len(kp) > 21:
                kp = kp[:21]

            if not kp:
                kp = [[0.0, 0.0] for _ in range(21)]
            processed_frames.append(kp)

        seq_len = len(processed_frames)

        if seq_len < self.target_frames:
            padding = [[[0.0, 0.0] for _ in range(21)] for _ in range(self.target_frames - seq_len)]
            processed_frames.extend(padding)
        elif seq_len > self.target_frames:
            processed_frames = processed_frames[:self.target_frames]

        tensor_data = torch.tensor(processed_frames, dtype=torch.float32)
        tensor_data = tensor_data.view(self.target_frames, -1)

        return tensor_data, torch.tensor(label_idx, dtype=torch.long)