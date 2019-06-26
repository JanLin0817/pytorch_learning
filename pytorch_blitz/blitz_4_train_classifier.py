import os
import copy
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter 

import time
import cv2
import glob
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize from [-1,1] bakc to [0,1]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_dataset(trainloader, classes):
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

class Net_only_by_nn(nn.Module):
    def __init__(self):
        super(Net_only_by_nn, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net_add_layer(nn.Module):
    '''
    All architecture are same as Net_only_by_nn except we add fc4 layer

    '''
    def __init__(self):
        super(Net_add_layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x
        
class Net_reduce_layer(nn.Module):
    '''
    All architecture are same as Net_only_by_nn except we add fc4 layer

    '''
    def __init__(self):
        super(Net_reduce_layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Net_diff_architecture(nn.Module):
    def __init__(self):
        super(Net_diff_architecture, self).__init__()
        self.conv1 = nn.Conv2d(4, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_CIFAR10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')                                             

    return trainloader, testloader, classes

def inference(testloader, net, device, classes):
    # net.to(device)
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))    

def save_model(model, save_dir='./models'):
    output = os.path.join(save_dir, 'classifier_net.pth')
    # torch.save(model.state_dict(), output)
    torch.save(model.state_dict(), output)

def load_pretrain_2_dict(strict=False, load_path='./models'):
    path = os.path.join(load_path, 'classifier_net.pth')
    state_dict = torch.load(path, map_location="cpu")
    return state_dict

def main():
    '''
    This is a training classifier example with
    1. training proces
    2. save model parameters as state_dict

    '''
    ## tensorboard writer
    tensorboard_path = 'runs/train_classifier'
    writer = SummaryWriter(tensorboard_path, comment='train')

    # setting
    gpu_id = 0
    resume = False
    if resume:
        resume_path = sorted(glob.glob('./models/*'))[-1]

    # 1. load dataset
    trainloader, testloader, classes = load_CIFAR10()
    print('INFO: training data {}'.format(len(trainloader)))
    
    # optional: show dataset
    if False:
        show_dataset(trainloader, classes)

    # create GPU device
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    print('INFO: get {}'.format(device))

    # 2. load network
    net = Net()
    net = nn.DataParallel(net)

    resume_epoch = 0
    if resume:
        resume_info = torch.load(resume_path)
        resume_epoch = resume_info['epoch']
        net.load_state_dict(resume_info['state_dict'])
    net.to(device) 

    # 3. define loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 4. start training
    for epoch in range(resume_epoch,10):  # loop over the whole dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # if i == 2000 + 1:
            #     break
            # get the inputs
            inputs, labels = data
            # on GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item() # loss.item() transfer tensot to python scalar
            
            if i % 2000 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                writer.add_scalar('running_loss/batch', running_loss / 2000, 8*i + len(trainloader)*epoch*8) # we use batch size as 8
                writer.add_scalar('running_loss/epoch', running_loss / 2000, epoch)
                # writer.add_graph(net, inputs)
                running_loss = 0.0

                save_checkpoint={
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                }
                output = os.path.join('./models', 'classifier_net' + '_epoch_' + str(epoch) + '_' + str(i) + '.pth')
                # torch.save(save_checkpoint, output)
    writer.close()

    print('INFO: Finished Training')    
    print('INFO: Saving model')

    save_model(net)
    # print images
    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    inference(testloader, net, device, classes)

def modify_architecture():
    net = Net_diff_architecture()

    # examine weight and bias in conv1 and conv2 before loading pretrained model
    conv1_weight_before_load = copy.deepcopy(net.conv1.weight)
    conv1_bias_before_load = copy.deepcopy(net.conv1.bias)
    conv2_weight_before_load = copy.deepcopy(net.conv2.weight)
    conv2_bias_before_load = copy.deepcopy(net.conv2.bias)    

    # load state_dict
    state_dict = load_pretrain_2_dict()

    # print(net.conv1.bias)
    # print(state_dict['conv1.bias'])
    # print(net.fc3.bias)
    # print(state_dict['fc3.bias'])   

    # modfiy state_dict's key adapt new arhcitecture
    state_dict['conv1.weight.ori'] = state_dict.pop('conv1.weight')
    #state_dict['conv1.bias.ori'] = state_dict.pop('conv1.bias')
    state_dict['fc3.weight.ori'] = state_dict.pop('fc3.weight')
    state_dict['fc3.bias.ori'] = state_dict.pop('fc3.bias')

    # load state_dict to model
    net.load_state_dict(state_dict, strict=False)

    # change initial weight for net.conv1.weight
    net.conv1.weight[:,:3,:,:] = nn.Parameter(state_dict['conv1.weight.ori'])
    nn.init.xavier_uniform_(net.conv1.weight)


    # examine weight and bias in conv1 and conv2 after loading pretrained model
    conv1_weight_after_load = net.conv1.weight
    conv1_bias_after_load = net.conv1.bias
    conv2_weight_after_load = net.conv2.weight
    conv2_bias_after_load = net.conv2.bias

    # compare the weight
    print('If conv1 weight before and after are the same: ', torch.equal(conv1_weight_before_load, conv1_weight_after_load))
    print('If conv1 bias before and after are the same: ', torch.equal(conv1_bias_before_load, conv1_bias_after_load))    
    print('If conv2 weight before and after are the same: ', torch.equal(conv2_weight_before_load, conv2_weight_after_load))
    print('If conv2 bias before and after are the same: ', torch.equal(conv2_bias_before_load, conv2_bias_after_load))    

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)

if __name__ == "__main__": 


    model_list = [Net, Net_only_by_nn, Net_add_layer, Net_reduce_layer, Net_diff_architecture]
    main()
    exit()

    ## Test network with different architecture
    modify_architecture()
    net = Net_diff_architecture()
    state_dict = load_pretrain_2_dict(net)

    # replace whole model's weight
    # net.apply(weights_init)
    exit()
    print('type of conv1 weight from state_dict {}'.format(type(state_dict['fc3.weight'])))
    print('type of conv1 weight from net {}'.format(type(net.conv1.weight)))
    
    # print the net weight, bias name and size
    net_key = list(net.state_dict().keys())
    state_dict_key = list(state_dict.keys())
    net_tensor_shape = list([item.size() for key, item in net.state_dict().items()])
    state_dict_tensor_shape = list([item.size() for key, item in state_dict.items()])

    print(net_key)
    print(state_dict_key)
    print(net_tensor_shape)
    print(state_dict_tensor_shape)

    # for key, item in net.state_dict().items():
    #     print(key, '\t', item.size())

    # for key, item in state_dict.items():
    #     print(key, '\t', item.size())

