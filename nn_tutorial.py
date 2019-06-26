import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x)
        return x
    
class Net_all_nn(nn.Module):

    def __init__(self):
        super(Net_all_nn, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool_2d = nn.MaxPool2d(kernel_size=(2,2))
        self.max_pool_2d_1 = nn.MaxPool2d(kernel_size=2)
        self.net = nn.Sequential(self.conv1, self.relu, self.max_pool_2d, self.conv2, self.relu, self.max_pool_2d)#, self.resize, self.fc1, self.relu, self.fc2, self.relu, self.fc3)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.relu(x)
        # x = self.max_pool_2d(x)
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.max_pool_2d(x)
        
        x = self.net(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        return x


    # def forward(self, x):
    #     # Max pooling over a (2, 2) window
    #     x = self.max_pool_2d(self.relu(self.conv1(x)))
    #     # If the size is a square you can only specify a single number
    #     x = self.max_pool_2d_1(self.relu(self.conv2(x)))
    #     x = x.view(-1, self.num_flat_features(x))
    #     x = self.relu(self.fc1(x))
    #     x = self.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def nn_test():
    conv2d = nn.Conv2d(1, 6, 5)
    # batch, channel, H, W
    x = torch.ones(10, 1, 64, 32, dtype=torch.float)
    print(x.size())
    x = conv2d(x)
    print(x.size())
    x = F.max_pool2d(F.relu(x), (2, 2))
    print('x size {}'.format(x.size()))
    x = x.view(-1, num_flat_features(x))
    print(x.size())
    print(type(conv2d))    

def unsqueeze():
    # unsqueeze a non-batch input as mini-batch input
    x = torch.ones(3, 64, 32, dtype=torch.float)
    print('\ninput size before unsqueeze: {}'.format(x.size()))
    x = x.unsqueeze(0)
    print('input size after unsqueeze: {}'.format(x.size()))

def instantiate_network(show=False):
    # instantiate network
    net = Net()
    if show:
        print('\n{}'.format(net)) 
        print(Net.mro()) # show it's super mro
    return net

def net_para(net):
    # use nn.parameters to show trainable parameter information
    print('\n=== net paramter ===\n')
    params = list(net.parameters())
    print('length of Net parameters: {}'.format(len(params)))
    print("conv1's weight: {}".format(params[0].size()))  # conv1's .weight

    print('each layers size')
    for i, para in enumerate(params):
        print('layer {}: paramter size: {}'.format(i, para.size()))

def calculate_loss(net):
    print('\n=== calculate loss ===\n')
    # test net with a input(fake channel one 32*32 image)
    in1 = torch.randn(1, 1, 32, 32)
    # caculate loss
    output = net(in1)
    target = torch.randn(10)  # a dummy target, for example
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss() # there are other loss function to pickup

    loss = criterion(output, target)
    print('loss: {}'.format(loss))
    return loss

def trace_graph(loss):
    # trace graph
    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

def backprop(net, loss):
    net.zero_grad()     # zeroes the gradient buffers of all parameters

    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward()

    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)    

def backprop_loop(net):
    # import input test net with a input(fake channel one 32*32 image)
    input = torch.randn(1, 1, 32, 32)

    # import targrt(ground truth)
    target = torch.randn(10)  # a dummy target, for example
    target = target.view(1, -1)  # make it the same shape as output

    # create loss function
    criterion = nn.MSELoss() 

    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # in your training loop:
    for i in range(100):
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()    # Does the update    
        print('INFO: step {}: loss={}'.format(i, loss))

if __name__ == "__main__":
    input = torch.randn(1, 1, 32, 32)
    target = torch.randn((1,10))
    net = Net()
    net.zero_grad() 
    # net = Net_all_nn()
    output = net(input)

    loss_func = nn.MSELoss()
    loss = loss_func(output, target)
    print(loss)

    conv1_grad = net.conv1.weight.grad
    print('conv1.bias.grad before backward')
    print(net.conv1.weight.grad)
d
    loss.backward()

    print('conv1.bias.grad after backward')
    print(net.conv1.weight.grad)

    print(net.modules())

    # for data in net.children():
    #     print(type(data))

    # for data in net.modules():
    #     print(type(data))

    # unsqueeze()
    # net = instantiate_network(show=False)
    # net_para(net)
    # loss = calculate_loss(net)
    # backprop(net, loss)
    # backprop_loop(net)