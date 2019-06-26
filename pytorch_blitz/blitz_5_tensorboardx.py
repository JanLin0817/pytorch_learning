import cv2
import time
import numpy as np

import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter 

if __name__ == "__main__":

    time_info = [str(x) for x in time.localtime()]
    # tensorboard_path = os.path.join('runs', '.'.join(time_info[:5]))
    tensorboard_path = 'runs/test_1'
    writer = SummaryWriter(tensorboard_path, comment='test')

    print(writer.add_graph)
    exit()
    # write in to tensorboard
    img_path = '/home/riq/segmentation/pytorch/data/faces/1084239450_e76e00b7e7.jpg'
    img = cv2.imread(img_path, 1)
    img = np.transpose(img, (2, 0, 1))
    img = (img - img.min()) / (img.max() - img.min())

    # tensor
    tensor = np.random.rand(120)
    tensor = torch.from_numpy(tensor)
    # exit()
    # tesnorboard
    for i in range(10):
        writer.add_image('eval_result', img, i)
        writer.add_image('test_result', img, i)
        writer.add_scalar('myscalar', i*2, i)
        writer.add_scalar('learning_rate', i*0.5, i)
