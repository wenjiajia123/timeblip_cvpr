import torch
import torch.nn as nn

# 定义输入维度和参数
input_channels = 1
output_channels = 2
kernel_size = 10
stride = 1
padding = 0

# 创建一维卷积层
conv1d = nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding)

# 创建输入张量
input_tensor = torch.randn(32, 75, 1)  # 维度为 [32, 1, 75]

# 将输入张量形状调整为适应一维卷积层的输入要求
input_tensor = input_tensor.transpose(1, 2)  # 将维度 1 和 2 交换，变为 [32, 75, 1]

# 应用一维卷积层
output_tensor = conv1d(input_tensor)

# 将输出张量形状调整回原始形状
import ipdb; ipdb.set_trace()
output_tensor = output_tensor.transpose(1, 2)  # 将维度 1 和 2 交换，变为 [32, 1, 75, 2]