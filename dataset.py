from torch.utils.data import Dataset
import numpy as np
import torch
import torchaudio


class FaceDataset(Dataset):
    def __init__(self, files_path):
        self.files_path = files_path

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, idx):
        file_path = self.files_path[idx]
        feature, label = xxxx()
        one_hot_label = torch.zeros(3)
        one_hot_label[label] = 1
        return (
            feature,
            one_hot_label,
        )
