from torch.utils.data import Dataset
import torch
from feature_engineering import return_feature


class FaceDataset(Dataset):
    def __init__(self, files_path):
        self.files_path = files_path

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, idx):
        file_path = self.files_path[idx]
        feature, label = return_feature(file_path)
        one_hot_label = torch.zeros(3)
        return (
            torch.tensor(feature),
            # torch.tensor(one_hot_label),
            torch.tensor(label)
        )
