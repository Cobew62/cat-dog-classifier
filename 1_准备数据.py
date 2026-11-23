"""
第 1 步：准备数据
============================================================
把猫狗图片整理成 PyTorch 能直接读取的目录结构：

    data/
      train/         训练集（让模型学习用的图片）
        cat/         猫的图片
        dog/         狗的图片
      val/           验证集（检验模型学得好不好的图片）
        cat/
        dog/

运行方式：
    打开命令行，进入本项目目录，执行：
        python 1_准备数据.py

脚本会把微软官方猫狗数据集压缩包解压、筛选并整理好上面的结构。
微软原包共 25000 张图，CPU 训练太慢，这里每类只取 2000 张：
1600 张训练 + 400 张验证，足够练出 95%+ 的准确率且速度快。
如果想用全部图片，把下面的 MAX_PER_CLASS 改成 None 即可。
"""

import os
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
ZIP_PATH = os.path.join(ROOT, "kagglecatsanddogs.zip")
MAX_PER_CLASS = 2000  # 每类取多少张图；None 表示用全部

# 微软数据集压缩包内图片所在目录
SRC_CAT = "PetImages/Cat"
SRC_DOG = "PetImages/Dog"


def is_image(name):
    return name.lower().endswith((".jpg", ".jpeg", ".png"))


def organize():
    """解压并从微软数据包中筛选图片，整理成标准目录结构。"""
    if not os.path.exists(ZIP_PATH):
        print(f"找不到数据集压缩包：{ZIP_PATH}")
        print("请先下载数据集，放到项目目录，文件名保持 kagglecatsanddogs.zip")
        raise SystemExit(1)

    print("正在读取数据集压缩包（文件较大，请稍候）...")
    z = zipfile.ZipFile(ZIP_PATH, "r")

    # 列出压缩包里所有猫狗图片路径
    cat_files = [n for n in z.namelist()
                 if n.startswith(SRC_CAT + "/") and is_image(n) and not n.endswith("/")]
    dog_files = [n for n in z.namelist()
                 if n.startswith(SRC_DOG + "/") and is_image(n) and not n.endswith("/")]
    print(f"压缩包中：猫 {len(cat_files)} 张，狗 {len(dog_files)} 张")

    if not cat_files or not dog_files:
        print("压缩包结构不符合预期，检查是否下载完整。")
        raise SystemExit(1)

    # 打乱后每类取指定数量（固定 seed 保证每次结果一致）
    import random
    random.seed(42)
    random.shuffle(cat_files)
    random.shuffle(dog_files)
    if MAX_PER_CLASS:
        cat_files = cat_files[:MAX_PER_CLASS]
        dog_files = dog_files[:MAX_PER_CLASS]

    # 划分：80% 训练，20% 验证
    split_at = int(len(cat_files) * 0.8)
    plan = {
        ("train", "cat"): cat_files[:split_at],
        ("val", "cat"): cat_files[split_at:],
        ("train", "dog"): dog_files[:split_at],
        ("val", "dog"): dog_files[split_at:],
    }

    # 清空旧数据，重新写入
    for (split, animal) in plan:
        folder = os.path.join(DATA_DIR, split, animal)
        os.makedirs(folder, exist_ok=True)
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))

    print("正在解压并整理图片...")
    for (split, animal), files in plan.items():
        folder = os.path.join(DATA_DIR, split, animal)
        saved = 0
        for src in files:
            name = os.path.basename(src)
            # 有些图片扩展名是大写 .JPG，统一存成 .jpg
            dst_name = f"{animal}_{saved}.jpg"
            try:
                data = z.read(src)
                # 跳过损坏或空文件
                if len(data) < 1000:
                    continue
                with open(os.path.join(folder, dst_name), "wb") as f:
                    f.write(data)
                saved += 1
            except Exception:
                continue
        print(f"  data/{split}/{animal}: {saved} 张图片")

    z.close()
    print(f"\n数据已整理到：{DATA_DIR}")


def show_summary():
    """打印最终的数据统计，方便确认数据没问题。"""
    print("\n数据目录统计：")
    total = 0
    for split in ("train", "val"):
        for animal in ("cat", "dog"):
            folder = os.path.join(DATA_DIR, split, animal)
            n = len([f for f in os.listdir(folder) if is_image(f)])
            total += n
            print(f"  data/{split}/{animal}: {n} 张")
    print(f"总计：{total} 张图片")

    if total == 0:
        print("警告：没有找到任何图片！")
    else:
        print("\n数据准备完成！下一步运行：python 2_训练.py")


if __name__ == "__main__":
    organize()
    show_summary()
