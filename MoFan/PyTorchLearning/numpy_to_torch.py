from traceback import print_tb
from turtle import dot
import torch
import numpy as np

np_data = np.arange(6).reshape((2,3))

#numpy转换为torch
torch_data = torch.from_numpy(np_data)

#torch转换为numpy
tensor_array = torch_data.numpy()
print(
    '\nnp_data',np_data,
    '\ntorch_data',torch_data,
    '\ntensor_array',tensor_array
)


data = [[1,2], [3,4]]

tensor = torch.FloatTensor(data)
data = np.array(data)
print(
    '\nnumpy:',data.dot(data),
    '\ntorch:',tensor.dot(tensor)
)