# License: BSD

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def show_state_dict(model=None):
    # Initialize model
    model = TheModelClass()

    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("Model's state_dict:")
    for para_name, para_tensor in model.state_dict().items():
        print(para_name, "\t", para_tensor.size(), '\t', type(para_tensor))

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name, var_tensor in optimizer.state_dict().items():
        print(var_name, "\t", var_tensor, '\t', type(para_tensor))

if __name__ == "__main__":
    show_state_dict()
    exit()
    path = '/home/riq/segmentation/DEXTR-PyTorch/models/' + 'MS_DeepLab_resnet_trained_VOC.pth'
    path = '/home/riq/segmentation/DEXTR-PyTorch/models/' + 'dextr_pascal-sbd.pth'
    deeplab_v2 = torch.load(path)
    for key, item in deeplab_v2.items():
        print(key, '\t', item.size())