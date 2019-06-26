import os
import torch
import numpy as np
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

def print_4_smaple(example_num=4):
    face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                            root_dir='data/faces/',
                                            )
    for i, sample in enumerate(face_dataset):

        print(i, sample['image'].shape, sample['landmarks'].shape)

        ax = plt.subplot(1, example_num, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_landmarks(**sample)

        if i == 3:
            plt.show()
            break

def show_data_transforms():
    face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                            root_dir='data/faces/',
                                            )
    scale = Rescale(256)
    crop = RandomCrop(224)
    composed = transforms.Compose([Rescale(256), RandomCrop(224)])

    # Apply each of the above transforms on sample.
    fig = plt.figure()
    sample = face_dataset[65]
    for i, tsfrm in enumerate([scale, crop, composed]):
        transformed_sample = tsfrm(sample)

        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        show_landmarks(**transformed_sample)

    plt.show()    

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].values
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

def data_loading_processing_flow():
    '''
    self-define data transform, self-define dataset, pytorch dataloader
    '''
    # 1. data transforms
    sample_transforms = transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()])

    # 2. prepare data set
    transformed_face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                            root_dir='data/faces/',
                                            transform=sample_transforms
                                            )                      
    # 3. data loader
    dataloader = DataLoader(transformed_face_dataset, batch_size=1,
                        shuffle=True, num_workers=4)

    # Now we can iterate from dataloader
    for i, sample in enumerate(dataloader):
        print(i, sample['image'].shape, sample['landmarks'].shape)

        if i > 3:
            break 

if __name__ == "__main__":
    sample_transforms = transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()])
    transformed_face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                            root_dir='data/faces/',
                                            transform=sample_transforms
                                            )                      

    dataloader = DataLoader(transformed_face_dataset, batch_size=1,
                        shuffle=True, num_workers=4)

    for i, sample in enumerate(dataloader):
        print(i, sample['image'].shape, sample['landmarks'].shape)

        if i > 3:
            break