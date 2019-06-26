import torch
import numpy as np
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,4]])

def numpy_torch_array():
    np_data = np.arange(6).reshape((2, 3))
    torch_data = torch.from_numpy(np_data)
    tensor2array = torch_data.numpy()
    print(
        '\nnumpy array:\n', np_data,          # [[0 1 2], [3 4 5]]
        '\ntorch tensor:\n', torch_data,      #  0  1  2 \n 3  4  5    [torch.LongTensor of size 2x3]
        '\ntensor to array:\n', tensor2array, # [[0 1 2], [3 4 5]]
    )

def tensor_mul():
    data = [[1,2], [3,4]]
    tensor = torch.FloatTensor(data)  # 转换成32位浮点 tensor
    # correct method
    print(
        '\nmatrix multiplication (matmul)',
        '\nnumpy: ', np.matmul(data, data),     # [[7, 10], [15, 22]]
        '\ntorch: ', torch.mm(tensor, tensor)   # [[7, 10], [15, 22]]
    )

    tensor_1D = torch.FloatTensor([1,2,3])
    ## !!!!  wrong way !!!!
    data = np.array(data)
    print(
        '\nmatrix multiplication (dot)\n',
        '\nnumpy: \n', data.dot(data),          # [[7, 10], [15, 22]] 在numpy 中可行
        '\ntorch: \n', tensor_1D.dot(tensor_1D) # torch only work for 1D tensor
    )   

def variable():
    # requires_grad involve gradient or not
    variable = Variable(tensor, requires_grad=True)
    print(variable)
    t_out = torch.mean(tensor*tensor)       # x^2
    v_out = torch.mean(variable*variable)   # x^2
    print(tensor*tensor)
    print(t_out)
    print(v_out)    # 7.5    

if __name__ == "__main__":
    x = torch.empty(5, 3, dtype=torch.double)      # new_* methods take in sizes
    print(x)
    print(x.size())

    x = torch.randn_like(x, dtype=torch.float)    # override dtype!
    print(x)                                      # result has the same size    
    print(x.size())
    pass