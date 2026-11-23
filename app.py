"""
猫狗分类网页应用
============================================================
用 Flask 搭建的网页服务：在浏览器上传一张图片，立刻告诉你它是猫还是狗。

运行方式：
    python app.py
然后在浏览器打开：http://localhost:5000

按 Ctrl+C 可以停止服务。
"""

import os
import io
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from torchvision import transforms, models
from PIL import Image

# ---------------- 基础配置 ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "cat_dog_model.pth")
IMAGE_SIZE = 224
CLASSES = ["cat", "dog"]                       # 顺序必须和训练时一致
CLASS_NAMES_ZH = {"cat": "猫", "dog": "狗"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 限制上传图片最大 16MB


# ---------------- 加载模型（服务启动时加载一次，之后复用）----------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()

# 用 weights=None 只搭建网络结构，权重完全来自我们自己训练好的模型文件，
# 这样启动时不需要联网下载预训练权重，部署到任何环境都能直接跑。
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print(f"模型加载成功，使用设备：{device}")

# 图片预处理，和训练/预测脚本保持一致
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------- 路由 ----------------
@app.route("/")
def index():
    """首页：显示上传界面。"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """接收上传的图片，返回预测结果（JSON）。"""
    # 检查请求里有没有文件
    if "file" not in request.files:
        return jsonify({"error": "没有收到图片文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "没有选择图片"}), 400

    try:
        # 读取上传的图片并转成 RGB
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 预处理 + 预测
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)

        # 取出概率最高的类别
        confidence, predicted_idx = probabilities.max(1)
        label = CLASSES[predicted_idx.item()]
        conf = confidence.item()
        cat_prob = probabilities[0][0].item()
        dog_prob = probabilities[0][1].item()

        return jsonify({
            "label": label,
            "name_zh": CLASS_NAMES_ZH[label],
            "confidence": round(conf * 100, 2),
            "cat_prob": round(cat_prob * 100, 2),
            "dog_prob": round(dog_prob * 100, 2),
        })
    except Exception as e:
        return jsonify({"error": f"图片处理失败：{str(e)}"}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("猫狗分类网页应用已启动")
    print("请在浏览器打开：http://localhost:5000")
    print("按 Ctrl+C 停止服务")
    print("=" * 50)
    app.run(host="127.0.0.1", port=5000, debug=False)
