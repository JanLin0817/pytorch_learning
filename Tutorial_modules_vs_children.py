import torch
import torch.nn as nn
from networks import resnet

net = resnet.resnet101(pretrained=False)
print(net)

# return same hierarchy as net 
for i, layer in enumerate(net.children()):
    print(i, layer)

# return expaand hierarchy of net 
for i, layer in enumerate(net.modules()):
    print(i, layer)

# return layer sequentially which no layer duplicate with nn.Sequential
layer_cnt = 0
for i, layer in enumerate(net.modules()):
    if not isinstance(layer, nn.Sequential) and not isinstance(layer, resnet.Bottleneck):
        layer_cnt = layer_cnt + 1
        print(i, layer)
print('total layer {}'.format(layer_cnt))