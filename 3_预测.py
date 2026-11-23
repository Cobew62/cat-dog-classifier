"""
第 3 步：用训练好的模型预测新图片
============================================================
加载训练好的模型，传入一张猫或狗的图片，输出它的判断和置信度。

运行方式：
    方式一：预测单张图片
        python 3_预测.py 你的图片.jpg

    方式二：不传参数，脚本会预测 data/val 文件夹里的前几张图片
        python 3_预测.py
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "cat_dog_model.pth")
IMAGE_SIZE = 224

# 类别顺序必须和训练时一致（ImageFolder 按文件夹名排序：cat 在前，dog 在后）
CLASSES = ["cat", "dog"]
CLASS_NAMES_ZH = {"cat": "猫", "dog": "狗"}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(device):
    """搭建和训练时一模一样的模型结构，再载入训练好的参数。"""
    try:
        from torchvision.models import ResNet18_Weights
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"找不到模型文件：{MODEL_PATH}")
        print("请先运行 python 2_训练.py 训练模型。")
        sys.exit(1)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # 评估模式，关闭随机性
    return model


def preprocess(image_path):
    """把一张图片处理成模型需要的输入格式。"""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # 加一个批次维度，形状变成 (1, 3, 224, 224)


@torch.no_grad()
def predict(model, image_path, device):
    """预测单张图片，返回类别和置信度。"""
    input_tensor = preprocess(image_path).to(device)
    outputs = model(input_tensor)
    probabilities = torch.softmax(outputs, dim=1)  # 转成概率
    confidence, predicted_idx = probabilities.max(1)

    label = CLASSES[predicted_idx.item()]
    return label, confidence.item()


def main():
    device = get_device()
    model = build_model(device)

    # 从命令行读取图片路径；如果没传，就自动找几张验证集图片做演示
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
    else:
        val_dir = os.path.join(ROOT, "data", "val")
        image_paths = []
        for animal in ("cat", "dog"):
            folder = os.path.join(val_dir, animal)
            if os.path.isdir(folder):
                files = sorted(
                    os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                )
                image_paths.extend(files[:2])  # 每类取 2 张
        if not image_paths:
            print("没有找到可预测的图片，请传入图片路径：python 3_预测.py 你的图片.jpg")
            return
        print("未指定图片，自动演示验证集中的几张图片：")

    print("-" * 50)
    for path in image_paths:
        if not os.path.exists(path):
            print(f"跳过不存在的文件：{path}")
            continue
        label, confidence = predict(model, path, device)
        name = os.path.basename(path)
        print(f"{name}: 预测 = {CLASS_NAMES_ZH[label]} ({label}), 置信度 = {confidence * 100:.2f}%")


if __name__ == "__main__":
    main()
