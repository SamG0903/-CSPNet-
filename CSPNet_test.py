import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义基本块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out

# 定义 CSPNet 块（简化）
class CSPNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPNetBlock, self).__init__()
        self.split_channels = out_channels // 2
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, self.split_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.split_channels),
            nn.ReLU(inplace=True),
            *[BasicBlock(self.split_channels, self.split_channels) for _ in range(num_blocks)]
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, self.split_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.split_channels),
            nn.ReLU(inplace=True),
            *[BasicBlock(self.split_channels, self.split_channels) for _ in range(num_blocks)]
        )
        
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.final_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        out = torch.cat((branch1, branch2), dim=1)
        out = self.final_conv(out)
        out = self.final_bn(out)
        return out

# 定义 CSPNet 模型
class CSPNet(nn.Module):
    def __init__(self, num_classes):
        super(CSPNet, self).__init__()
        self.initial_conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)  # 修改通道数
        self.initial_bn = nn.BatchNorm2d(32)
        self.initial_relu = nn.ReLU(inplace=True)

        self.csp_block1 = CSPNetBlock(32, 64, num_blocks=1)  # 减少数量
        self.csp_block2 = CSPNetBlock(64, 128, num_blocks=1)  # 减少数量

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)  # 更新输入通道数

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        
        x = self.csp_block1(x)
        x = self.csp_block2(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # 扁平化
        x = self.fc(x)
        return x

# 创建模型实例
model = CSPNet(num_classes=10).to(device)  # 假设有 10 个分类

# 模拟数据集
batch_size = 8  # 减少批量大小
num_samples = 200  # 减少样本数量
inputs = torch.randn(num_samples, 3, 112, 112)  # 改为 112x112
targets = torch.randint(0, 10, (num_samples,), dtype=torch.long)  # 随机生成目标标签

# 创建数据加载器
train_dataset = TensorDataset(inputs, targets)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
total_loss=0
for epoch in range(5):  # 减少训练轮次
    model.train()  # 设置模型为训练模式
    total_loss = 0  # 初始化总损失
    for inputs, targets in train_loader:  # 迭代数据加载器中的每个批次
        inputs, targets = inputs.to(device), targets.to(device)  # 移动数据到设备
        optimizer.zero_grad()  # 清零梯度
        
        outputs = model(inputs)  # 计算输出
        loss = criterion(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss = total_loss+loss.item()  # 累加损失
    print(f"Epoch [{epoch+1}/5], Loss: {total_loss/len(train_loader):.4f}")

# 输出的形状
outputs = model(inputs[:batch_size])  # 运行前向传播并使用前 8 个样本
print("Output shape:", outputs.shape)  # 应该是 [8, 10]



