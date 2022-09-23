import numpy as np
import torch
import random

# k = 2
# tensor = torch.rand(2, 10)
# print(tensor)
# values, indices = torch.topk(torch.abs(tensor.data), k=k)
# print(values)
# print(indices)
#
# values = tensor.data[indices]
# print(values)

# tensor = torch.rand(100).reshape(2, 2, 5, 5)
# print(tensor)
# temp = tensor.reshape(1, -1)
# print(temp)
# k = int(temp.numel() * 0.1)
# values, indices = torch.topk(temp, k, largest=True)
# print(values)
# boundary = float(values[:, -1])
# print(boundary)
# mask = torch.ge(tensor, boundary)
# print(mask)
# result = mask * tensor
# print(result)

# mask = torch.zeros_like(tensor)
# values, indices = torch.topk(torch.abs(tensor.data), k=k)
# print(indices)
# mask = torch.gt(tensor, 10)
# print(mask)
# mask = torch.nonzero(mask)
# print(mask)

# 二维tensor
# for i in range(len(mask)):
#     for j in range(len(mask[0])):
#         if j in indices[i]:
#             mask[i][j] = 1
# print(mask)

# 四维tensor
# for m in range(len(mask)):
#     for n in range(len(mask[0])):
#         for i in range(len(mask[0][0])):
#             for j in range(len(mask[0][0][0])):
#                 if j in indices[m][n][i]:
#                     mask[m][n][i][j] = 1
# print(mask)



# mask = torch.zeros_like(tensor)
# for i, param in enumerate(tensor):
#     values, indices = torch.topk(torch.abs(param.data), k=k)
#     for j in range(tensor.numel()):
#         mask[i][j] = 1 if j in indices.numpy() else 0
#     print(mask[i])


# tensor = torch.rand(2, 3, dtype=torch.float32)
# print(tensor)
# xq = torch.quantize_per_tensor(tensor, scale=0.0036, zero_point=0, dtype=torch.quint8)
# print(xq)
# xdq = torch.dequantize(xq)
# print(xdq)

# tensor = torch.from_numpy(np.random.choice(2, 20)).reshape(2, 2, 1, 5)
# print(tensor)
# num = float(torch.count_nonzero(tensor))
# print(num)
#
# tensor = torch.rand(10).reshape(2, 5)
# print(tensor)
# print(tensor.reshape(-1, 1))
# print(tensor)

# list = []
# result = [0] * 20
# for i in range(20):
#     list.append(i)
#     # print(list)
#
#     # if i > 5 and len([j for j in list[-6:] if j > 5]) == 5:
#     #     result[i] = 1
#     if i >= 4:
#         print([j for j in list[-5:]])

# print(result)

tensor1 = torch.rand(10).reshape(2, 5)
tensor2 = torch.rand(10).reshape(2, 5)

temp1 = torch.quantize_per_tensor(tensor1, scale=0.005, zero_point=0, dtype=torch.qint8)
temp2 = torch.quantize_per_tensor(tensor2, scale=0.005, zero_point=0, dtype=torch.qint8)
# temp = temp1 + temp2  # error
# print(temp)