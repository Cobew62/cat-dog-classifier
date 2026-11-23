"""
第 2 步：训练模型
============================================================
用迁移学习训练一个猫狗分类器：
  1. 加载在 ImageNet 上预训练好的 ResNet-18（已经学会了看边缘、纹理、形状）
  2. 冻结它的大部分参数（不再改动，保留通用识别能力）
  3. 把最后的输出层换成只区分「猫 / 狗」两类
  4. 用我们的猫狗图片训练这个新的输出层

运行方式：
    python 2_训练.py

训练完成后，模型会保存到 models/cat_dog_model.pth，
并画出准确率曲线 models/训练曲线.png。
"""

import os
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")  # 无界面环境下也能保存图片
import matplotlib.pyplot as plt

# ---------------- 可调整的参数（新手可以先不动）----------------
BATCH_SIZE = 32       # 每次喂给模型多少张图片
EPOCHS = 5            # 把全部数据反复学习几遍
LEARNING_RATE = 1e-4  # 学习步长，太大容易学不好，太小学得慢
IMAGE_SIZE = 224      # 图片统一缩放到 224x224

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "cat_dog_model.pth")


def get_device():
    """自动选择可用的设备：优先显卡，其次苹果芯片，最后用 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms():
    """定义图片的预处理：缩放、增强、转成张量、标准化。"""
    # 训练集加上随机翻转和旋转，相当于"数据增强"，让模型见更多样的图片
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        # 用 ImageNet 的均值和标准差标准化，和预训练模型保持一致
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # 验证集不需要随机增强，保持图片原样才能公平评估
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def load_data():
    """用 ImageFolder 读取图片，文件夹名字自动变成类别标签。"""
    # 先检查数据是否准备好，给出更友好的提示
    for split in ("train", "val"):
        for animal in ("cat", "dog"):
            folder = os.path.join(DATA_DIR, split, animal)
            if not os.path.isdir(folder) or not os.listdir(folder):
                print(f"数据缺失或为空：{folder}")
                print("请先运行：python 1_准备数据.py")
                raise SystemExit(1)

    train_transform, val_transform = build_transforms()
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), val_transform)

    # Windows 上多进程可能报错，num_workers 设为 0 最稳妥
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"类别对应关系：{train_dataset.class_to_idx}（0=cat 猫, 1=dog 狗）")
    print(f"训练图片：{len(train_dataset)} 张，验证图片：{len(val_dataset)} 张")
    return train_loader, val_loader


def build_model(device):
    """加载预训练 ResNet-18，冻结主体，替换最后的分类层为 2 类输出。"""
    # 新版 torchvision 用 weights 参数，老版用 pretrained=True，这里做兼容
    try:
        from torchvision.models import ResNet18_Weights
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(pretrained=True)

    # 冻结所有卷积层的参数：它们已经学会了通用的图像特征，不需要再训练
    for param in model.parameters():
        param.requires_grad = False

    # 替换最后的全连接层：输入特征数不变，输出改成 2（猫、狗）
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    model = model.to(device)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一轮，返回这一轮的平均损失和准确率。"""
    model.train()  # 切换到训练模式
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)              # 前向传播：模型做出预测
        loss = criterion(outputs, labels)    # 计算预测和真实答案的差距

        optimizer.zero_grad()  # 清空上一次的梯度
        loss.backward()        # 反向传播：算出每个参数该怎么调整
        optimizer.step()       # 更新参数

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """在验证集上测试，不更新参数。"""
    model.eval()  # 切换到评估模式
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def plot_curve(history):
    """把训练过程画成曲线图并保存。"""
    epochs = range(1, len(history["train_acc"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], "o-", label="train acc")
    plt.plot(epochs, history["val_acc"], "o-", label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Training accuracy curve")
    plt.legend()
    plt.grid(alpha=0.3)
    path = os.path.join(MODEL_DIR, "训练曲线.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"准确率曲线已保存：{path}")


def main():
    device = get_device()
    print(f"使用设备：{device}")
    print("-" * 50)

    train_loader, val_loader = load_data()
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    # 只把需要训练的参数（也就是新分类层）交给优化器
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    history = {"train_acc": [], "val_acc": []}
    best_acc = 0.0

    print("开始训练...")
    for epoch in range(EPOCHS):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"第 {epoch + 1}/{EPOCHS} 轮 | "
              f"训练准确率 {train_acc * 100:.2f}% | "
              f"验证准确率 {val_acc * 100:.2f}% | "
              f"耗时 {time.time() - start:.1f} 秒")

        # 只保存验证准确率最高的那一版模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    print("-" * 50)
    print(f"训练完成！最佳验证准确率：{best_acc * 100:.2f}%")
    print(f"模型已保存：{MODEL_PATH}")
    plot_curve(history)
    print("\n下一步运行：python 3_预测.py")


if __name__ == "__main__":
    main()
