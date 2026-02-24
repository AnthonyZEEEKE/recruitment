# 导入需要的工具，不用改
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 固定随机种子，保证结果可复现，不用改
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

# 自动判断用显卡还是CPU，不用改
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"你的运行设备: {device}")

# ---------------------- 1. 数据加载（自动下载数据集，不用改） ----------------------
# 数据预处理，不用改
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 自动下载CIFAR-10数据集，不用改
train_dataset_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# 划分训练集和验证集，不用改
train_size = int(0.9 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

# 数据加载器，CPU跑不动就把batch_size改成32/16，其他不用改
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# CIFAR-10的10个类别，不用改
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ---------------------- 2. 搭建神经网络（不用改） ----------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 卷积层，提取图片特征
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # 全连接层，做分类
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# 模型初始化，不用改
model = SimpleCNN().to(device)

# ---------------------- 3. 训练配置（不用改） ----------------------
# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 训练轮次，着急的话改成50，也能达标，100轮效果更好
num_epochs = 100

# 用来记录训练过程的指标，不用改
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

# ---------------------- 4. 开始训练（不用改） ----------------------
print("===== 开始训练，不用操作，等进度条走完 =====")
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    train_bar = tqdm(train_loader, desc=f"第{epoch+1}/{num_epochs}轮 训练")
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计指标
        train_loss += loss.item() * images.size(0)
        _, predict = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predict == labels).sum().item()
        
        train_bar.set_postfix(loss=loss.item(), 准确率=100*train_correct/train_total)
    
    # 计算本轮训练的平均指标
    train_loss_epoch = train_loss / train_total
    train_acc_epoch = 100 * train_correct / train_total

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"第{epoch+1}/{num_epochs}轮 验证")
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predict = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predict == labels).sum().item()
            
            val_bar.set_postfix(loss=loss.item(), 准确率=100*val_correct/val_total)
    
    val_loss_epoch = val_loss / val_total
    val_acc_epoch = 100 * val_correct / val_total

    # 保存本轮指标
    train_loss_list.append(train_loss_epoch)
    val_loss_list.append(val_loss_epoch)
    train_acc_list.append(train_acc_epoch)
    val_acc_list.append(val_acc_epoch)

    # 打印本轮结果
    print(f"第{epoch+1}轮完成 | 训练准确率:{train_acc_epoch:.2f}% | 验证准确率:{val_acc_epoch:.2f}%\n")

# ---------------------- 5. 测试集最终评估（不用改） ----------------------
print("===== 训练完成，开始测试集评估 =====")
model.eval()
test_correct = 0
test_total = 0
# 保存所有结果用来可视化
all_images = []
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predict = torch.max(outputs, 1)
        
        test_total += labels.size(0)
        test_correct += (predict == labels).sum().item()
        
        all_images.append(images.cpu())
        all_preds.append(predict.cpu())
        all_labels.append(labels.cpu())

# 计算最终测试集准确率
test_acc = 100 * test_correct / test_total
print(f"===== 最终测试集准确率: {test_acc:.2f}% =====")
print(f"只要这个数字大于50%，就完成了及格线要求！")

# 拼接所有结果
all_images = torch.cat(all_images)
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

# ---------------------- 6. 自动生成可视化图片（必交内容，不用改） ----------------------
# 图片反归一化，用来正常显示
def denormalize(img):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)
    img = img * std + mean
    return img.clamp(0,1)

# 1. 生成训练Loss和准确率曲线
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(num_epochs), train_loss_list, label="训练Loss")
plt.plot(range(num_epochs), val_loss_list, label="验证Loss")
plt.xlabel("训练轮次")
plt.ylabel("Loss值")
plt.title("Loss变化曲线")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1,2,2)
plt.plot(range(num_epochs), train_acc_list, label="训练准确率")
plt.plot(range(num_epochs), val_acc_list, label="验证准确率")
plt.xlabel("训练轮次")
plt.ylabel("准确率(%)")
plt.title("准确率变化曲线")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("task3_训练曲线.png", dpi=300)
plt.close()
print("已生成：task3_训练曲线.png")

# 2. 生成分类正确的示例图片
plt.figure(figsize=(10,5))
correct_idx = (all_preds == all_labels).nonzero().squeeze()
plt.suptitle("分类正确示例", fontsize=14)
for i in range(5):
    idx = correct_idx[i]
    img = denormalize(all_images[idx])
    true_label = classes[all_labels[idx]]
    pred_label = classes[all_preds[idx]]
    
    plt.subplot(1,5,i+1)
    plt.imshow(img.permute(1,2,0))
    plt.title(f"真实:{true_label}\n预测:{pred_label}", fontsize=10)
    plt.axis("off")
plt.tight_layout()
plt.savefig("task3_正确分类示例.png", dpi=300)
plt.close()
print("已生成：task3_正确分类示例.png")

# 3. 生成分类错误的示例图片
plt.figure(figsize=(10,5))
wrong_idx = (all_preds != all_labels).nonzero().squeeze()
plt.suptitle("分类错误示例", fontsize=14)
for i in range(5):
    idx = wrong_idx[i]
    img = denormalize(all_images[idx])
    true_label = classes[all_labels[idx]]
    pred_label = classes[all_preds[idx]]
    
    plt.subplot(1,5,i+1)
    plt.imshow(img.permute(1,2,0))
    plt.title(f"真实:{true_label}\n预测:{pred_label}", fontsize=10)
    plt.axis("off")
plt.tight_layout()
plt.savefig("task3_错误分类示例.png", dpi=300)
plt.close()
print("已生成：task3_错误分类示例.png")

# 保存模型权重
torch.save(model.state_dict(), "task3_model.pth")
print("===== 所有文件已生成，任务三核心内容完成！=====")