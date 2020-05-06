# from __future__ import print_function
# import torch

# x = torch.tensor([5, 3], dtype=torch.float64)
# if torch.cuda.is_available():
#     device = torch.device("cuda")          # a CUDA device object
#     y = torch.ones_like(x, device="cuda")  # 直接在GPU上创建tensor
#     x = x.to("cuda")                       # 或者使用`.to("cuda")`方法
#     z = x + y
#     print(y)
#     print(x)
#     print(z)
#     print(z.to("cpu", torch.double))       # `.to`也能在移动时改变dtype

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 2x2 Max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果是方阵,则可以只使用一个数字进行定义
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
# 网络结构
# print(net)

# 网络参数
# params = list(net.parameters())
# print(len(params))
# print(params[0].size())  # conv1's .weight

# 网络输出
# input = torch.randn(1, 1, 32, 32)
# output = net(input)

# # 计算损失
# target = torch.randn(10)  # 本例子中使用模拟数据
# target = target.view(1, -1)  # 使目标值与数据值形状一致
# criterion = nn.MSELoss()
# loss = criterion(output, target)
# print(loss)
# print(loss.grad_fn)  # MSELoss
# print(loss.grad_fn.next_functions[0][0])  # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# print(net.conv1.bias.grad)  # None
# net.zero_grad()     # 清零所有参数(parameter）的梯度缓存

# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)  # None

# loss.backward()

# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)

# 更新权重
# learning_rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)

import torch.optim as optim

# 创建优化器(optimizer）
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 在训练的迭代中：
optimizer.zero_grad()   # 清零梯度缓存
input = torch.randn(1, 1, 32, 32)
output = net(input)

target = torch.randn(10)  # 本例子中使用模拟数据
target = target.view(1, -1)  # 使目标值与数据值形状一致
criterion = nn.MSELoss()
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 更新参数