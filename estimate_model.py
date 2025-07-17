import torch
import json
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.metrics import auc, f1_score, roc_curve, classification_report, confusion_matrix
from itertools import cycle
from numpy import interp
import torch.nn as nn
from matplotlib.ticker import LogFormatterMathtext

# 全局路径配置
PATHS = {
    'confusion_matrix': './results/confusion_matrix',
    'f1score': './results/f1score',
    'loss': './results/loss',
    'roc': './results/roc',
    'output': './output',
    'class_indices': './class_indices',
    'test_results': './test_results'
}

# 初始化所有目录
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

def load_class_indices():
    """加载类别索引文件"""
    # 直接在当前目录查找
    json_path = 'classes_indices.json'
    
    # 如果找不到，尝试在 class_indices 目录中查找
    if not os.path.exists(json_path):
        json_path = os.path.join('class_indices', 'classes_indices.json')
    
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading class indices: {e}")
        print(f"尝试的路径: {json_path}")
        exit(1)

@torch.no_grad()
def Plot_ROC(net, val_loader, save_name, device):
    try:
        json_file = open('./classes_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    score_list = []
    label_list = []

    net.load_state_dict(torch.load(save_name))

    for i, data in enumerate(val_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = torch.softmax(net(images), dim=1)
        score_tmp = outputs
        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())

    score_array = np.array(score_list)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], len(class_indict.keys()))
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)

    print("score_array:", score_array.shape)  # (batchsize, classnum)
    print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])

    # 调用sklearn库，计算每个类别对应的fpr和tpr
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(len(class_indict.keys())):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(len(class_indict.keys()))]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(len(set(label_list))):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])

    # Finally average it and compute AUC
    mean_tpr /= len(class_indict.keys())
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

    # 绘制所有类别平均的roc曲线
    plt.figure(figsize=(12, 12))
    lw = 2

    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(class_indict.keys())), colors):
        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(class_indict[str(i)], roc_auc_dict[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw, label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('./multi_classes_roc.png')
    # plt.show()

def data_transformation():
    """统一的数据预处理方法"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

@torch.no_grad()
def predict_single_image(model, device, weights, page_size=9):
    data_transform = data_transformation()

    # read class_indict
    json_path = './classes_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # load model weights
    assert os.path.exists(weights), "weight file does not exist."
    checkpoint = torch.load(weights, map_location=device)
    if 'state_dict' in checkpoint:  # 如果检查点包含完整训练状态
        state_dict = checkpoint['state_dict']
    else:  # 如果检查点直接保存的是模型参数
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    # Path to the test images
    path = '/Users/cosmicdawn/Downloads/ViT/newdata/12-multi'
    testList = os.listdir(path)

    # Ensure the path exists
    assert os.path.exists(path), "file: '{}' does not exist.".format(path)

    total_images = len(testList)
    total_pages = (total_images + page_size - 1) // page_size  # 计算总页数

    for page in range(total_pages):
        fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(40, 20))  # 每行6个子图，3个图片和3个对应的直方图
        axes = axes.flatten()
        for ax in axes:
            ax.axis('off')

        # Display images and histograms for the current page
        for idx in range(page_size):
            image_idx = page * page_size + idx
            if image_idx >= total_images:
                break

            file = testList[image_idx]
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue

            try:
                img = Image.open(os.path.join(path, file))
                img = img.convert('RGB')
                # img = img.transpose(Image.FLIP_TOP_BOTTOM)  # 随机翻转图片
                img_transformed = data_transform(img)
                img_transformed.unsqueeze_(0)  # 添加批次维度

                # Predict
                output = torch.squeeze(model(img_transformed.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                sorted_probs, sorted_indices = torch.sort(predict, descending=True)
                
                # 获取前两名的类别和概率
                top1_idx = sorted_indices[0].item()
                top2_idx = sorted_indices[1].item()
                top1_prob = sorted_probs[0].item()
                top2_prob = sorted_probs[1].item()

                class_probs = {class_indict[str(i)]: prob.item() for i, prob in enumerate(predict)}

                # Prepare text for image label and probability
                class_text = "Class: {}".format(class_indict[str(top1_idx)])
                prob_text = "Prob: {:.3}".format(top1_prob)

                # 判断是否为多标签
                if top1_prob - top2_prob < 0.1:
                    # 如果差值小于0.1，则认为是多标签图像
                    class_text = "Class: {} & {}".format(class_indict[str(top1_idx)], class_indict[str(top2_idx)])
                    prob_text = "Probs: {:.3}, {:.3}".format(top1_prob, top2_prob)
                    print(f"Multi-label image detected: {class_text}")
                
                # Show the picture
                img_ax = axes[idx * 2]  # 每个图片在偶数索引
                img_ax.imshow(img)
                img_ax.set_title(f"{class_text}\n{prob_text}", fontsize=20)
                img_ax.axis('off')  # 保持图片显示

                # Plot histogram of probabilities
                hist_ax = axes[idx * 2 + 1]  # 每个直方图在奇数索引
                class_names = list(class_probs.keys())
                probs = list(class_probs.values())
                hist_ax.barh(class_names, probs, color='skyblue')  # 显示类别名称和概率
                hist_ax.set_xscale('log')
                hist_ax.set_xlim([1e-4, 1])  # 设置x轴的范围
                hist_ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1])  # 设置刻度
                hist_ax.get_xaxis().set_major_formatter(LogFormatterMathtext())  # 使用科学计数法格式
                hist_ax.set_xlabel("Probability", fontsize=16)
                hist_ax.set_title("Class Probabilities", fontsize=16)
                hist_ax.tick_params(axis='y', labelsize=14)  # 调整y轴类别标签字体大小
                hist_ax.axis('on')
                # 在每个条形上添加百分比文本
                for i, (prob, class_name) in enumerate(zip(probs, class_names)):
                    hist_ax.text(prob + 0.01, i, f'{prob * 100:.1f}%', va='center', fontsize=12)  # 显示百分比，位置调整可微调
                print(f"Image: {file}")
                for class_name, prob in class_probs.items():
                    print(f"Class: {class_name}, Probability: {prob:.3f}")

            except Exception as e:
                print(f"处理文件 '{file}' 时出错: {e}")

        plt.tight_layout()
        plt.show()


@torch.no_grad()
def Predictor(net, test_loader, save_name, device, fold):
    # 加载类别信息
    class_indict = load_class_indices()
    
    # 初始化数据结构
    y_pred, y_true = [], []

    # 加载模型
    net.load_state_dict(torch.load(save_name))
    net.eval()

    # 收集预测结果
    for images, labels in test_loader:
        images = images.to(device)
        preds = torch.argmax(torch.softmax(net(images), dim=1), dim=1)
        y_pred.extend(preds.cpu().tolist())
        y_true.extend(labels.cpu().tolist())

    # 计算指标
    accuracy = 100 * np.mean(np.array(y_pred) == np.array(y_true))
    f1 = 100 * f1_score(y_true, y_pred, average='weighted')
    
    # 保存混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 8))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.savefig(os.path.join(PATHS['confusion_matrix'], f'fold{fold}_cm.png'))
    plt.close()

    # 保存分类报告
    report = classification_report(y_true, y_pred, target_names=list(class_indict.values()), output_dict=True)
    report_path = os.path.join(PATHS['f1score'], f'fold{fold}_report.json')
    with open(report_path, 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report
        }, f, indent=2)

    print(f"Fold {fold} results saved to {report_path}")

