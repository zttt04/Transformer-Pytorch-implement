import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，防止图形窗口弹出
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from Transformer_model import Transformer
from config import TransformerConfig
from mask import create_masks
"""
文件说明：
本脚本用于训练基于Transformer架构的机器翻译模型（英文→德文）。
主要包含以下核心功能：

1. 训练流程：
   - 使用PyTorch实现完整的训练循环
   - 支持多GPU训练（通过CUDA自动检测）
   - 包含梯度裁剪防止梯度爆炸
   - 实现Transformer论文中的学习率调度策略

2. 数据管理：
   - 加载预处理后的索引数据（en/de processed_indexes.pt）
   - 使用DataLoader进行批量处理和数据打乱
   - 支持自定义批大小（config.batch_size）

3. 模型配置：
   - 通过config.TransformerConfig集中管理超参数
   - 支持自定义层数、头数、维度等核心参数
   - 使用Xavier初始化权重

4. 训练监控：
   - 实时显示训练进度条（tqdm）
   - 记录每200个batch的损失值和学习率
   - 自动绘制并保存损失曲线（位于fig目录）

5. 模型保存：
   - 每个epoch结束后自动保存模型权重
   - 使用绝对路径确保跨平台兼容性
   - 权重文件按epoch编号命名（Weight目录）

6. 日志系统：
   - 记录每个epoch的训练摘要（epoch_training_logs.csv）
   - 保存详细的batch级训练日志（batch_training_logs.csv）
   - 自动创建目录结构（logs目录）

输出结果：
- 模型权重文件（位于Weight目录）
- 训练损失曲线（fig/transformer_loss_final1.png）
- 详细训练日志（logs目录）
"""
# 读取模型配置参数
config = TransformerConfig()

# 训练参数设置
batch_size = config.batch_size       # 批处理大小
epochs = config.epochs               # 训练轮数
warmup_steps = config.warmup_steps   # Transformer论文中的学习率warmup步数

# 定义Transformer专用的学习率调度器
class TransformerLRScheduler:
    def __init__(self, d_model, warmup_steps):
        self.d_model = d_model          # 模型维度（论文中默认512）
        self.warmup_steps = warmup_steps  # 学习率warmup阶段步数
        self.step_num = 0               # 当前训练步数计数器

    def step(self, optimizer):
        # 按照论文公式更新学习率
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -1.5))
        # 更新优化器学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr  # 返回当前学习率，方便日志记录

# 绘制训练损失曲线的函数
def plot_loss(steps, losses, save_path):
    plt.figure(figsize=(10, 6))  # 设置画布尺寸
    plt.plot(steps, losses, linewidth=1.5, color='blue')  # 绘制损失曲线
    plt.xlabel('Batch', fontsize=14)  # x轴标签
    plt.ylabel('Loss', fontsize=14)  # y轴标签
    plt.title('Loss every 200 batches', fontsize=16)  # 图表标题
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)  # 添加网格线
    plt.savefig(save_path)  # 保存图表到指定路径
    plt.close()  # 关闭图表释放内存

# 主训练函数
def train():
    # 设备选择（优先使用GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取当前脚本所在的项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 加载预处理后的索引数据
    en_idx = torch.load(os.path.join(project_root, 'DataProcessing', 'Train', 'en_processed_indexes.pt'))
    de_idx = torch.load(os.path.join(project_root, 'DataProcessing', 'Train', 'de_processed_indexes.pt'))

    # 创建数据集和数据加载器
    dataset = TensorDataset(en_idx, de_idx)  # 将英文和德文数据配对
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True  # 每个epoch前打乱数据
    )

    # 初始化Transformer模型并移动到目标设备
    model = Transformer(TransformerConfig()).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=1)  # 忽略填充索引（假设1为填充标记）
    optimizer = optim.Adam(
        model.parameters(),
        betas=(0.9, 0.98),  # 论文推荐的Adam参数
        eps=1e-9
    )

    # 初始化学习率调度器
    scheduler = TransformerLRScheduler(
        d_model=TransformerConfig().d_model,
        warmup_steps=warmup_steps
    )

    # 初始化训练日志记录变量
    losses = []        # 记录每200个batch的损失值
    steps = []         # 记录对应的训练步数
    step_count = 0     # 当前总训练步数计数器

    # 初始化详细日志数据存储
    batch_log_data = []  # 存储每200个batch的日志
    epoch_log_data = []  # 存储每个epoch的日志

    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        train_loss = 0.0  # 初始化epoch总损失

        # 使用tqdm创建进度条
        batch_iterator = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}",
            leave=False  # 不在控制台保留进度条
        )

        for src_batch, tgt_batch in batch_iterator:
            # 将数据移动到目标设备
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            # 创建源序列和目标序列的掩码
            src_mask, tgt_mask = create_masks(
                src_batch,
                tgt_batch[:, :-1],  # 目标序列去掉最后一个token（用于teacher forcing）
                1,  # 填充标记索引
                device
            )

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            output = model(src_batch, tgt_batch[:, :-1], src_mask, tgt_mask)

            # 计算损失
            loss = criterion(
                output.view(-1, output.size(-1)),  # 展平预测结果
                tgt_batch[:, 1:].contiguous().view(-1)  # 展平真实标签（跳过第一个token）
            )

            # 反向传播
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)

            # 更新模型参数
            optimizer.step()

            # 更新学习率
            current_lr = scheduler.step(optimizer)

            # 累加损失值
            train_loss += loss.item()

            # 更新进度条显示
            batch_iterator.set_postfix(
                loss=loss.item(),
                lr=f"{current_lr:.8f}"  # 显示当前学习率
            )

            # 记录每200个batch的训练信息
            step_count += 1
            if step_count % 200 == 0:
                losses.append(loss.item())
                steps.append(step_count)
                # 绘制并保存损失曲线
                plot_loss(
                    steps,
                    losses,
                    os.path.join(project_root, 'fig', 'transformer_loss_final1.png')
                )

                # 记录详细日志
                batch_log_data.append({
                    'Epoch': epoch + 1,
                    'Batch': step_count,
                    'Loss': loss.item(),
                    'Learning Rate': current_lr
                })

        # 计算epoch平均损失
        epoch_loss = train_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{epochs} Loss: {epoch_loss:.4f}')

        # 保存模型权重
        save_path = os.path.join(project_root, 'Weight', f'transformer_epoch_{epoch + 1}_final1.pth')
        torch.save(model.state_dict(), save_path)
        print(f"权重已保存: {save_path}")

        # 记录epoch级别的日志
        epoch_log_data.append({
            'Epoch': epoch + 1,
            'Epoch Loss': epoch_loss,
            'Final Learning Rate': current_lr
        })

        # 保存epoch级别的日志到CSV
        epoch_csv_save_path = os.path.join(project_root, 'logs', 'epoch_training_logs_final1.csv')
        os.makedirs(os.path.dirname(epoch_csv_save_path), exist_ok=True)
        with open(epoch_csv_save_path, 'w', newline='') as csvfile:
            fieldnames = ['Epoch', 'Epoch Loss', 'Final Learning Rate']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(epoch_log_data)
        print(f"Epoch {epoch + 1} 训练日志已保存到 {epoch_csv_save_path}")

    # 保存batch级别的详细日志到CSV
    batch_csv_save_path = os.path.join(project_root, 'logs', 'batch_training_logs_final1.csv')
    os.makedirs(os.path.dirname(batch_csv_save_path), exist_ok=True)
    with open(batch_csv_save_path, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Batch', 'Loss', 'Learning Rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(batch_log_data)
    print(f"每200个batch训练日志已保存到 {batch_csv_save_path}")

# 主程序入口
if __name__ == "__main__":
    train()