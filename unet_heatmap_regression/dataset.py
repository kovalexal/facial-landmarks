import os

import cv2
import torch
from torch.utils import data
import numpy as np
import pandas as pd


class ThousandLandmarksDataset(data.Dataset):
    def __init__(self, root, transforms, split="train", TRAIN_SIZE=0.8):
        super(ThousandLandmarksDataset, self).__init__()
        self.root = root
        landmark_file_name = os.path.join(root, 'landmarks.csv') if split is not "test" \
            else os.path.join(root, "test_points.csv")
        images_root = os.path.join(root, "images")

        self.image_names = []
        self.landmarks = []

        # Получим число строк в файле
        header = None
        with open(landmark_file_name, "rt") as fp:
            header = fp.readline()
            num_lines = 1 + sum(1 for line in fp)
        num_lines -= 1  # header
        header_types = {k: np.int16 if k != 'file_name' else str for k in header.split('\t')}

        # Прочитаем train часть, или test часть
        if split == 'train':
            landmark_info = pd.read_csv(landmark_file_name, sep='\t', dtype=header_types, nrows=int(num_lines * TRAIN_SIZE))
        elif split == 'val':
            landmark_info = pd.read_csv(landmark_file_name, sep='\t', dtype=header_types, skiprows=lambda x: (x > 0) and (x < 1 + int(num_lines * TRAIN_SIZE)))
        else:
            landmark_info = pd.read_csv(landmark_file_name, sep='\t', dtype=header_types)

        # Добавляем название папки
        landmark_info['file_name'] = landmark_info['file_name'].apply(lambda x: os.path.join(images_root, x))
        self.image_names = landmark_info['file_name'].tolist()

        if split in ("train", "val"):
            # Читаем landmark'и
            self.landmarks = [x[1].values.reshape((int((landmark_info.shape[1] - 1) / 2), 2)).astype(np.int16) for x in landmark_info.drop(columns=['file_name']).iterrows()]
            self.landmarks = torch.from_numpy(np.stack(self.landmarks))
        else:
            self.landmarks = None

        self.transforms = transforms

    def __getitem__(self, idx):
        sample = {}
        if self.landmarks is not None:
            # landmarks = self.landmarks[idx]
            landmarks = self.landmarks[idx].clone()
            sample["landmarks"] = landmarks

        image = cv2.imread(self.image_names[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample["image"] = image

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.image_names)
