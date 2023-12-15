import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # 一个线性层，输入维度为10，输出维度为5
        self.relu = nn.ReLU()  # ReLU 激活函数

    def forward(self, x):
        x = self.fc1(x)  # 数据经过第一个线性层
        x = self.relu(x)  # 通过 ReLU 激活函数
        return x

# 创建一个模型实例
model = SimpleNet()

# 假设有一个输入张量
input_data = torch.randn(3, 10)  # 3个样本，每个样本具有10个特征

# 将输入数据传递给模型，调用 forward 方法
output = model(input_data)
print(output)
