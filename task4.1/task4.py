# 任务四：CIFAR-10模型优化+进阶指标分析（适配任务三环境，直接运行）
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 固定随机种子，保证结果可复现（和任务三一致）
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

# 自动适配GPU/CPU（和任务三一致，无需修改）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"运行设备: {device}")

# ---------------------- 1. 优化版数据增强（核心优化1：提升泛化能力） ----------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # 新增：随机旋转15°，增加图片多样性
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 新增：颜色抖动
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载数据集（复用任务三的data，download=False避免重复下载）
train_dataset_full = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

# 划分训练/验证集（和任务三一致：9:1）
train_size = int(0.9 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

# 自适应batch_size（GPU=128，CPU=32，自动匹配）
batch_size = 128 if torch.cuda.is_available() else 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# CIFAR-10类别（和任务三一致）
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ---------------------- 2. 优化版CNN模型（核心优化2：加深结构+批量归一化） ----------------------
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # 卷积层+批量归一化（BatchNorm2d）：加速收敛，稳定训练
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # 新增：批量归一化，解决梯度消失
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 新增：第四层卷积，加深特征提取
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # 全连接层：提高Dropout率，进一步缓解过拟合
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512),  # 适配新卷积层的输出维度
            nn.ReLU(),
            nn.Dropout(0.6),  # 任务三是0.5，新增：提高到0.6，减少过拟合
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平特征
        x = self.fc_layers(x)
        return x

# 初始化优化模型，放到指定设备
model = ImprovedCNN().to(device)

# ---------------------- 3. 优化训练策略（核心优化3：学习率衰减+早停） ----------------------
criterion = nn.CrossEntropyLoss()  # 损失函数和任务三一致
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器和任务三一致
# 新增：学习率衰减——每20轮学习率减半，前期快学，后期精调
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

num_epochs = 80  # 优化后收敛更快，80轮足够（比任务三100轮省时间）
best_val_acc = 0.0  # 早停专用：保存验证集准确率最高的模型
patience = 10  # 新增：早停——10轮验证集准确率没提升，直接停止训练，避免过拟合

# 记录训练指标（新增学习率记录）
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []
lr_list = []  # 记录每轮学习率变化

# ---------------------- 4. 开始训练（带早停，全程自动） ----------------------
print("===== 任务四：优化模型训练开始 =====")
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    train_bar = tqdm(train_loader, desc=f"第{epoch+1}/{num_epochs}轮 训练")
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, predict = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predict == labels).sum().item()
        
        train_bar.set_postfix(loss=loss.item(), 准确率=100*train_correct/train_total)
    
    # 计算训练指标
    train_loss_epoch = train_loss / train_total
    train_acc_epoch = 100 * train_correct / train_total
    
    # 验证阶段（和任务三一致，无梯度计算）
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
    
    # 学习率衰减生效
    scheduler.step()
    lr_list.append(optimizer.param_groups[0]['lr'])
    
    # 保存指标
    train_loss_list.append(train_loss_epoch)
    val_loss_list.append(val_loss_epoch)
    train_acc_list.append(train_acc_epoch)
    val_acc_list.append(val_acc_epoch)
    
    # 早停逻辑：保存最优模型，避免过拟合
    if val_acc_epoch > best_val_acc:
        best_val_acc = val_acc_epoch
        patience_counter = 0
        torch.save(model.state_dict(), "task4_best_model.pth")  # 保存最优模型权重
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"早停触发！第{epoch+1}轮验证集准确率未提升，停止训练")
            break  # 直接终止训练，节省时间
    
    print(f"第{epoch+1}轮完成 | 训练准确率:{train_acc_epoch:.2f}% | 验证准确率:{val_acc_epoch:.2f}% | 当前学习率:{lr_list[-1]:.6f}\n")

# ---------------------- 5. 测试集进阶评估（任务四核心：多指标分析） ----------------------
print("===== 加载最优模型，开始测试集进阶分析 =====")
model.load_state_dict(torch.load("task4_best_model.pth", map_location=device))
model.eval()

# 收集测试集所有结果（用于混淆矩阵/类别分析）
test_correct = 0
test_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predict = torch.max(outputs, 1)
        
        test_total += labels.size(0)
        test_correct += (predict == labels).sum().item()
        
        all_preds.extend(predict.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 最终测试集准确率
test_acc = 100 * test_correct / test_total
print(f"===== 任务四最终测试集准确率: {test_acc:.2f}% =====")

# 计算每个类别的单独准确率（任务四必分析指标）
class_correct = list(0. for _ in range(10))
class_total = list(0. for _ in range(10))
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# 打印每个类别准确率
class_acc = []
print("\n===== 每个类别的分类准确率 =====")
for i in range(10):
    acc = 100 * class_correct[i] / class_total[i]
    class_acc.append(acc)
    print(f"{classes[i]}: {acc:.2f}%")

# ---------------------- 6. 自动生成任务四所有可视化/文本文件（必交） ----------------------
# 反归一化函数（和任务三一致，保证图片正常显示）
def denormalize(img):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)
    img = img * std + mean
    return img.clamp(0,1)

# 图1：训练曲线+学习率变化（优化版，3个子图）
plt.figure(figsize=(15, 5))
# 子图1：Loss曲线
plt.subplot(1,3,1)
plt.plot(train_loss_list, label="训练Loss", color="#1f77b4")
plt.plot(val_loss_list, label="验证Loss", color="#ff7f0e")
plt.xlabel("训练轮次")
plt.ylabel("Loss值")
plt.title("Loss变化曲线（优化版）")
plt.legend()
plt.grid(alpha=0.3)
# 子图2：准确率曲线
plt.subplot(1,3,2)
plt.plot(train_acc_list, label="训练准确率", color="#1f77b4")
plt.plot(val_acc_list, label="验证准确率", color="#ff7f0e")
plt.xlabel("训练轮次")
plt.ylabel("准确率(%)")
plt.title("准确率变化曲线（优化版）")
plt.legend()
plt.grid(alpha=0.3)
# 子图3：学习率衰减曲线
plt.subplot(1,3,3)
plt.plot(lr_list, color="#2ca02c")
plt.xlabel("训练轮次")
plt.ylabel("学习率")
plt.title("学习率衰减曲线")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("task4_训练曲线_优化版.png", dpi=300)
plt.close()
print("✅ 已生成：task4_训练曲线_优化版.png")

# 图2：混淆矩阵（任务四核心进阶指标，展示类别间误分情况）
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('预测类别', fontsize=12)
plt.ylabel('真实类别', fontsize=12)
plt.title('CIFAR-10混淆矩阵（优化版模型）', fontsize=14)
plt.tight_layout()
plt.savefig("task4_混淆矩阵.png", dpi=300)
plt.close()
print("✅ 已生成：task4_混淆矩阵.png")

# 图3：类别准确率柱状图（直观展示各品类表现）
plt.figure(figsize=(12, 6))
bars = plt.bar(classes, class_acc, color='skyblue', edgecolor='black')
plt.xlabel('类别', fontsize=12)
plt.ylabel('准确率(%)', fontsize=12)
plt.title('每个类别的分类准确率（优化版模型）', fontsize=14)
plt.ylim(0, 100)
# 柱子上标注具体数值
for bar, acc in zip(bars, class_acc):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{acc:.1f}%", ha='center')
plt.tight_layout()
plt.savefig("task4_类别准确率.png", dpi=300)
plt.close()
print("✅ 已生成：task4_类别准确率.png")

# 图4：错误分类案例（带真实/预测标签，分析模型短板）
plt.figure(figsize=(12, 6))
wrong_idx = np.where(np.array(all_preds) != np.array(all_labels))[0]
plt.suptitle("优化模型错误分类案例（真实标签/预测标签）", fontsize=14)
for i in range(6):
    idx = wrong_idx[i]
    img, _ = test_dataset[idx]
    img = denormalize(img)
    true_label = classes[all_labels[idx]]
    pred_label = classes[all_preds[idx]]
    plt.subplot(2, 3, i+1)
    plt.imshow(img.permute(1,2,0))
    plt.title(f"真实：{true_label}\n预测：{pred_label}", fontsize=10)
    plt.axis("off")
plt.tight_layout()
plt.savefig("task4_错误分类案例.png", dpi=300)
plt.close()
print("✅ 已生成：task4_错误分类案例.png")

# 生成分类报告（文本文件，含精准率/召回率/F1值，任务四必交）
with open("task4_分类报告.txt", "w", encoding="utf-8") as f:
    f.write("===== CIFAR-10分类报告（优化版模型）=====\n")
    f.write(classification_report(all_labels, all_preds, target_names=classes, digits=2))
print("✅ 已生成：task4_分类报告.txt")

# 保存最终模型权重
torch.save(model.state_dict(), "task4_final_model.pth")
print("✅ 已保存：task4_final_model.pth（最终模型权重）")

print("\n===== 任务四所有文件生成完成！=====")
print("📁 生成文件均在UD_task3文件夹内，直接用于提交！")