import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import torchvision.models as models
import time  # 导入时间模块

# ====== 配置 ======
model_path = '/home/ynao/data/newvit/output/resnoweight_acc0.93_epoch93_lr0.0001_loss0.8232424317336664.pth'
txt_path = 'test.txt'
num_classes = 11
device = torch.device("cuda")
print(f"CUDA available: {torch.cuda.is_available()}")  # 应该输出False
print(f"Device: {device}")  # 应该输出cpu
# ====== 数据预处理 ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ====== 加载模型 ======
print("正在加载模型...")
model_load_start = time.time()

# model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
# model.heads = torch.nn.Sequential(nn.Linear(768, num_classes))


model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(in_features=2048, out_features=11, bias=True)

#
# 加载检查点并提取模型权重
checkpoint = torch.load(model_path, map_location=device)
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.to(device)
model.eval()

model_load_time = time.time() - model_load_start
print(f"模型加载完成，耗时: {model_load_time:.4f} 秒")

# ====== 推理并收集标签 ======
true_labels = []
pred_labels = []
total_time = 0.0  # 总处理时间
inference_time = 0.0  # 纯推理时间
image_count = 0  # 处理的图片数量

with open(txt_path, 'r') as f:
    lines = f.readlines()
    total_lines = len(lines)
    print(f"开始处理 {total_lines} 张图片...")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) != 2:
            print(f"跳过无效行: {line}")
            continue
        path, label = parts
        label = int(label)

        # 记录开始时间
        start_time = time.time()

        # 加载和预处理图像
        image = Image.open(path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        # 记录推理开始时间
        inference_start = time.time()

        # 模型推理
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()

        # 记录推理结束时间
        inference_end = time.time()

        # 记录总结束时间
        end_time = time.time()

        # 更新时间统计
        total_time += (end_time - start_time)
        inference_time += (inference_end - inference_start)
        image_count += 1

        # 添加标签（从0-10变为1-11）
        true_labels.append(label + 1)
        pred_labels.append(pred + 1)

        # 每处理100张图片打印一次进度
        if image_count % 100 == 0:
            print(f"已处理 {image_count}/{total_lines} 张图片...")

# 计算平均时间
avg_total_time = total_time / image_count * 1000  # 转换为毫秒
avg_inference_time = inference_time / image_count * 1000  # 转换为毫秒

print("\n===== 时间统计 =====")
print(f"处理总图片数: {image_count}")
print(f"总处理时间: {total_time:.4f} 秒")
print(f"平均每张图片处理时间: {avg_total_time:.2f} 毫秒 (包含加载和预处理)")
print(f"平均每张图片推理时间: {avg_inference_time:.2f} 毫秒 (仅模型推理)")
print(f"模型推理占总时间比例: {(inference_time / total_time) * 100:.1f}%")

# ====== 计算并绘制混淆矩阵 ======
labels = list(range(1, num_classes + 1))
cm = confusion_matrix(true_labels, pred_labels, labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Classes 1-11)")
plt.tight_layout()
plt.show()

# ====== 性能指标 ======
precision = precision_score(true_labels, pred_labels, average='weighted')
recall = recall_score(true_labels, pred_labels, average='weighted')
f1 = f1_score(true_labels, pred_labels, average='weighted')

print("\n===== 分类性能指标 =====")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall:    {recall:.4f}")
print(f"Weighted F1 Score:  {f1:.4f}")

# 完整报告
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels,
                            labels=labels,
                            target_names=[str(i) for i in labels]))