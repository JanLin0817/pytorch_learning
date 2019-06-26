from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, VOCDetection

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, json_file, image_dir, transform=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.landmarks_frame = pd.read_json(json_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

def json_parser():
    f = pd.read_json('./obj_00001.json')
    cols = f.columns
    for col in cols:
        print(col, type(col))

    frames = f.loc[:,'frames']
    print('length of annotation {}'.format(len(frames)))    

if __name__ == "__main__":
    voc_dir 
    voc = VOCDetection()