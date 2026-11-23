# 猫狗识别器 🐱🐶

一个基于 **PyTorch 迁移学习（ResNet-18）** 的猫狗二分类项目，附带 **Flask 网页应用**。
上传一张图片，模型会告诉你它是猫还是狗，并给出两者的概率。

适合深度学习入门：代码全中文注释，CPU 即可训练，无需显卡。

---

## 1. 项目解决什么问题

给出一个图像分类的完整闭环：**准备数据 → 训练模型 → 命令行预测 → 网页应用**。

具体来说，它解决这几个问题：

- **快速上手图像分类**：用迁移学习代替从零训练，只需少量图片、普通电脑 CPU 就能在几分钟内得到一个高准确率模型（实测验证准确率 **96.38%**）。
- **学完能直接用**：训练脚本、预测脚本、网页界面一应俱全，既是教程也是可运行的项目。
- **从命令行到网页的完整体验**：命令行脚本适合学习原理，网页应用适合展示和分享给别人。

---

## 2. 主要功能

| 功能 | 说明 |
|------|------|
| 数据准备 | 从微软官方猫狗数据集压缩包中，每类抽取 2000 张图片，按 8:2 划分训练集 / 验证集，整理成 PyTorch `ImageFolder` 标准结构 |
| 模型训练 | 加载 ResNet-18 预训练架构，冻结特征提取层，只训练最后的二分类头；自动保存最优模型和准确率曲线 |
| 命令行预测 | 传入一张图片路径输出识别结果；不带参数时自动用验证集图片演示 |
| 网页应用 | Flask 服务 + 上传界面，支持点击 / 拖拽上传、图片预览、识别结果和概率条展示 |
| 设备自适应 | 自动选择 CUDA 显卡 / 苹果 MPS / CPU |

技术栈：Python · PyTorch · torchvision · Pillow · Flask · matplotlib

---

## 3. 安装方法

### 环境要求

- Python 3.8+（推荐 3.9～3.11，本项目在 3.13 上验证通过）
- 无需 GPU，CPU 即可运行；有 NVIDIA 显卡会自动加速

### 安装步骤

**① 克隆仓库并进入目录**

```bash
git clone https://github.com/Cobew62/cat-dog-classifier.git
cd cat-dog-classifier
```

**②（推荐）创建虚拟环境**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**③ 安装依赖**

```bash
pip install -r requirements.txt
```

依赖清单见 `requirements.txt`：torch、torchvision、Pillow、matplotlib、flask。
国内网络慢可加镜像源：

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

**④ 验证安装**

```bash
python -c "import torch, torchvision, flask; print('环境 OK')"
```

### 关于数据与模型

- **模型已随仓库提供**：`models/cat_dog_model.pth`（约 43MB），可以直接做预测和启动网页，**无需重新训练**。
- **训练数据不随仓库提供**：单张原始压缩包 787MB，超过 GitHub 单文件 100MB 限制。如需重新训练，请自行下载：

  > https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip

  下载后重命名为 `kagglecatsanddogs.zip` 放到项目根目录，再运行 `1_准备数据.py`。
  可用 `MAX_PER_CLASS = None` 使用全部 25000 张图片（训练时间会显著变长）。

---

## 4. 使用方法

### 方式 A：直接用网页应用（最简单）

适合只想体验识别、不想碰训练的人。

```bash
python app.py
```

浏览器打开 **http://localhost:5000**，点击或拖拽上传一张猫狗图片，点「开始识别」即可。
按 `Ctrl+C` 停止服务。命令行窗口保持打开，网页才能访问。

> 网页是本地服务，只有你自己电脑能访问。要分享给别人，需要部署到服务器或云平台。

### 方式 B：命令行预测

```bash
# 预测指定图片
python 3_预测.py 你的图片.jpg

# 不传参数，自动用验证集图片演示
python 3_预测.py
```

### 方式 C：从零训练模型

依次执行三步：

```bash
# 第 1 步：准备数据（需先放好数据集压缩包，见上文）
python 1_准备数据.py

# 第 2 步：训练模型，约 9 分钟（CPU）
python 2_训练.py

# 第 3 步：预测
python 3_预测.py 你的图片.jpg
```

训练参数在 `2_训练.py` 顶部修改：`BATCH_SIZE=32`、`EPOCHS=5`、`LEARNING_RATE=1e-4`。
训练完成后生成 `models/cat_dog_model.pth` 和 `models/训练曲线.png`。

---

## 5. 输入输出示例

### 命令行预测

输入：

```bash
python 3_预测.py
```

输出：

```
未指定图片，自动演示验证集中的几张图片：
--------------------------------------------------
cat_0.jpg: 预测 = 猫 (cat), 置信度 = 96.20%
dog_0.jpg: 预测 = 狗 (dog), 置信度 = 99.46%
```

### 网页应用

输入：在页面上传一张猫的图片 → 点击「开始识别」

输出：页面显示

```
识别结果
猫 (cat)
置信度 96.2%

🐱 猫  ████████████████████ 96.2%
🐶 狗  █ 3.8%
```

### HTTP 接口

网页背后的接口，可供其他程序调用。

请求：

```http
POST /predict
Content-Type: multipart/form-data

file: <图片二进制，字段名 file>
```

成功响应（JSON）：

```json
{
  "label": "cat",
  "name_zh": "猫",
  "confidence": 96.2,
  "cat_prob": 96.2,
  "dog_prob": 3.8
}
```

字段说明：

| 字段 | 含义 |
|------|------|
| `label` | 英文类别：`cat` 或 `dog` |
| `name_zh` | 中文类别：猫 / 狗 |
| `confidence` | 最高类别的置信度（百分比） |
| `cat_prob` | 判为猫的概率（百分比） |
| `dog_prob` | 判为狗的概率（百分比） |

错误响应：缺少文件时返回 `{"error": "..."}`，HTTP 400；图片处理失败返回 HTTP 500。

### 训练过程输出

```
使用设备：cpu
类别对应关系：{'cat': 0, 'dog': 1}（0=cat 猫, 1=dog 狗）
训练图片：3199 张，验证图片：800 张
开始训练...
第 1/5 轮 | 训练准确率 79.77% | 验证准确率 91.25% | 耗时 112.0 秒
...
第 5/5 轮 | 训练准确率 94.56% | 验证准确率 96.38% | 耗时 111.5 秒
训练完成！最佳验证准确率：96.38%
```

---

## 项目结构

```
cat-dog-classifier/
├── 1_准备数据.py          # 整理数据集
├── 2_训练.py              # 迁移学习训练
├── 3_预测.py              # 命令行预测
├── app.py                 # Flask 网页服务后端
├── templates/
│   └── index.html         # 网页前端界面
├── models/
│   ├── cat_dog_model.pth  # 训练好的模型（随仓库提供）
│   └── 训练曲线.png
├── requirements.txt       # 依赖清单
├── .gitignore
└── README.md
```

> `data/` 和 `kagglecatsanddogs.zip` 由 `1_准备数据.py` 生成，不随仓库提交。

---

## 模型说明

采用迁移学习：加载 ResNet-18 的网络结构，冻结其卷积特征提取层（保留通用的边缘、纹理识别能力），
仅把最后的全连接层替换为输出 2 类的分类头并用猫狗图片训练。
模型文件 `cat_dog_model.pth` 保存的就是这套参数，加载时 `weights=None` 只搭结构、不联网下载权重，
因此预测和网页服务可离线运行。

- 输入：RGB 图片，统一缩放到 224×224，按 ImageNet 均值/标准差标准化
- 输出：2 类 logits → softmax 得到猫/狗概率

---

## 常见问题

| 现象 | 解决办法 |
|------|----------|
| `python` 不是内部命令 | 重装 Python 并勾选 Add to PATH；或用 `py` 代替 `python` |
| 提示找不到模型文件 | 确认 `models/cat_dog_model.pth` 存在；或先跑 `2_训练.py` |
| 双击 html 文件报"获取失败" | 不能直接打开 html，必须 `python app.py` 后访问 http://localhost:5000 |
| 训练数据缺失 | 按第 3 节下载数据集压缩包并运行 `1_准备数据.py` |
| 准确率偏低（<70%） | 检查数据目录结构；把 `EPOCHS` 调大到 10 |
| 依赖安装失败 | 换用阿里云 / 清华镜像源 |

---

## 许可证

本项目采用 [MIT License](LICENSE)，欢迎自由使用、修改和学习。

Copyright (c) 2026 Cobew62
