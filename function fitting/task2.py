import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 数据加载
def load_data(file_path):
    df = pd.read_csv(file_path)
    x = df['x'].values.reshape(-1, 1)
    y = df['y'].values.reshape(-1, 1)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 2. 模型构建
class MLP(nn.Module):
    def __init__(self, hidden_layers=[64, 64, 64]):
        super(MLP, self).__init__()
        layers = []
        input_dim = 1
        
        # 添加隐藏层
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # 添加输出层
        layers.append(nn.Linear(input_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# 3. 模型训练
def train_model(model, x_train, y_train, epochs=1000, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    for epoch in range(epochs):
        # 前向传播
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # 每100个epoch打印一次loss
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return losses

# 4. 可视化对比
def plot_comparison(x, y, model, epoch, ax):
    x_np = x.numpy()
    y_np = y.numpy()
    
    # 生成预测值
    with torch.no_grad():
        y_pred = model(x).numpy()
    
    # 绘制真实数据散点图
    ax.scatter(x_np, y_np, s=10, label='True Data', alpha=0.5)
    
    # 绘制拟合曲线
    ax.plot(x_np, y_pred, 'r-', label=f'Fitted Curve (Epoch={epoch})')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Comparison at Epoch={epoch}')
    ax.legend()

# 5. 主函数
def main():
    # 加载数据
    file_path = 'task2.csv'
    x, y = load_data(file_path)
    
    # 分割训练集和测试集（用于判断过拟合）
    train_size = int(0.8 * len(x))
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 构建模型
    model = MLP(hidden_layers=[128, 128, 128])
    
    # 训练模型
    epochs = 1000
    lr = 0.001
    print(f'Training with learning rate: {lr}')
    losses = train_model(model, x_train, y_train, epochs=epochs, lr=lr)
    
    # 绘制Loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.savefig('loss_curve.png')
    plt.show()
    
    # 绘制不同Epoch时的拟合曲线
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 训练10个epoch
    model_10 = MLP(hidden_layers=[128, 128, 128])
    train_model(model_10, x_train, y_train, epochs=10, lr=lr)
    plot_comparison(x, y, model_10, 10, axes[0])
    
    # 训练100个epoch
    model_100 = MLP(hidden_layers=[128, 128, 128])
    train_model(model_100, x_train, y_train, epochs=100, lr=lr)
    plot_comparison(x, y, model_100, 100, axes[1])
    
    # 训练1000个epoch
    plot_comparison(x, y, model, 1000, axes[2])
    
    plt.tight_layout()
    plt.savefig('comparison_curves.png')
    plt.show()
    
    # 学习率分析：测试不同学习率
    print('\n=== Learning Rate Analysis ===')
    learning_rates = [0.00001, 0.001, 1.0]
    
    for lr in learning_rates:
        print(f'\nTraining with learning rate: {lr}')
        model_lr = MLP(hidden_layers=[128, 128, 128])
        losses_lr = train_model(model_lr, x_train, y_train, epochs=100, lr=lr)
        
        # 绘制Loss曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 101), losses_lr)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve with Learning Rate={lr}')
        plt.savefig(f'loss_curve_lr_{lr}.png')
        plt.show()
    
    # 过拟合判定：计算训练集和测试集的损失
    print('\n=== Overfitting Analysis ===')
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        train_loss = criterion(model(x_train), y_train).item()
        test_loss = criterion(model(x_test), y_test).item()
    
    print(f'Training Loss: {train_loss:.4f}')
    print(f'Test Loss: {test_loss:.4f}')
    
    if test_loss > train_loss * 1.2:
        print('Model may be overfitting!')
    else:
        print('Model is not overfitting.')

if __name__ == '__main__':
    main()
