import argparse
import torch.optim as optim
import torch
from torchvision import transforms
from PIL import Image
import csv
from torchtoolbox.transform import Cutout
from collections import Counter
import torchvision.models as models
import torch.nn as nn
from timm.utils import accuracy
from estimate_model import Predictor
import os
import matplotlib.pyplot as plt
import json



def get_args_parser():
    parser = argparse.ArgumentParser(
        'model training and evaluation script', add_help=False)
    parser.add_argument('--size', default=224, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_class',default=11)
    parser.add_argument('--model', choices=['vit', 'resnet'], default='vit',
                        help='choose which model to use: vit or resnet')
    parser.add_argument('--augment4x', choices=['aug4x', 'noaug'], default='noaug',
                    help='choose whether to use 4x data augmentation (aug4x or noaug)')

    parser.add_argument('--out_dir',default='./output')
    return parser

class MyTrainDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, transform=None, augment4x=False):
        self.file_list = file_list
        self.transform = transform
        self.augment4x = augment4x

    def __len__(self):
        return len(self.file_list) * 4 if self.augment4x else len(self.file_list)

    def __getitem__(self, idx):
        if self.augment4x:
            actual_idx = idx // 4
            rotation_idx = idx % 4
        else:
            actual_idx = idx
            rotation_idx = 0  

        img_path, label = self.file_list[actual_idx]
        image = Image.open(img_path).convert('RGB')

        if self.augment4x:
            if rotation_idx == 1:
                image = image.rotate(90)
            elif rotation_idx == 2:
                image = image.rotate(180)
            elif rotation_idx == 3:
                image = image.rotate(270)

        if self.transform:
            image = self.transform(image)

        return image, label


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_labels, transform=None):
        self.image_labels = image_labels
        self.transform = transform

    def __getitem__(self, index):
        image_path, label = self.image_labels[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_labels)


def load_image_labels(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        image_labels = [(line[0].replace('\\', '/'), int(line[2])) for line in reader]
    return image_labels


def save_class_indices(image_paths, labels, output_dir='./'):
    class_names = set([os.path.basename(os.path.dirname(path)) for path in image_paths])
    class_indices = {class_name: idx for idx, class_name in enumerate(sorted(class_names))}
    reversed_class_indices = {str(idx): class_name for class_name, idx in class_indices.items()}
    json_path = os.path.join(output_dir, 'classes_indices.json')
    with open(json_path, 'w') as json_file:
        json.dump(reversed_class_indices, json_file, indent=4)

    print(f'Class indices saved to {json_path}')

def train(model, train_loader, optimizer, args, criterion, epoch, train_losses):
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    scaler = torch.amp.GradScaler(device="cuda", enabled=True)  # 初始化一次即可
    print("total_num:", total_num, "len(train_loader):", len(train_loader))
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        scaler.scale(loss).backward()  
        scaler.step(optimizer)         
        scaler.update()                

        lr = optimizer.param_groups[0]['lr']
        sum_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f'Train Epoch: {epoch} [{(batch_idx + 1) * len(data):>5d}/{total_num} '
                  f'({100. * (batch_idx + 1) / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}\tLR: {lr:.9f}')

    ave_loss = sum_loss / len(train_loader)
    train_losses.append(ave_loss)
    print(f'epoch: {epoch}, loss: {ave_loss:.4f}')

def val(model, val_loader, args, criterion, epoch, val_losses, ACC, best_loss, best_ckpt_path, lr, model_name):

    model.eval()
    val_loss, correct = 0, 0
    total_num = len(val_loader.dataset)
    acc1_sum, acc5_sum = 0, 0  
    print("total_num:", total_num, "len(val_loader):", len(val_loader))

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            _, pred = torch.max(output.data, 1)
            correct += (pred == target).sum().item()
            val_loss += loss.item()

            acc1_sum += acc1.item()
            acc5_sum += acc5.item()

        acc = correct / total_num
        avgloss = val_loss / len(val_loader)
        val_losses.append(avgloss)
        acc1_avg = acc1_sum / len(val_loader)  
        acc5_avg = acc5_sum / len(val_loader)  

        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Acc@1: {:.2f}%, Acc@5: {:.2f}%\n'.format(
            avgloss, correct, total_num, 100 * acc, acc1_avg, acc5_avg))

        augment_tag = args.augment4x
        if (avgloss < best_loss or acc > ACC) and acc > 0.75:
            best_loss = avgloss 
            ACC = acc  
            ckpt_path = os.path.join(args.out_dir, f'{model_name}_{augment_tag}_acc{acc:.2f}_epoch{epoch}_lr{lr}_loss{best_loss}.pth')
            print("Saving model checkpoint to {}".format(ckpt_path))
            torch.save(model.state_dict(), ckpt_path)
            best_ckpt_path = ckpt_path  

    return ACC, best_loss, best_ckpt_path

def plot_losses(train_losses, val_losses, fold):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title(f'Loss Curves for Fold {fold}', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(loc='upper right')
    plt.savefig(f'./loss/fold{fold}_loss_curve.png')
    plt.close() 

def load_model(args):
    if args.model == 'vit':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.dropout = nn.Dropout(p=0.1)
        model.heads = nn.Sequential(nn.Linear(768, args.num_class))
        model_name = 'vit'
    else:  # resnet
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, args.num_class)
        model_name = 'resnet'

    return model.to(args.device), model_name

def main(args):
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # 分别加载训练集和验证集文件
    def load_split_data(file_path):
        with open(file_path, 'r') as f:
            return [(line.split(',')[0], int(line.split(',')[1])) for line in f]

    # 加载数据
    train_data = load_split_data('train.txt')
    val_data = load_split_data('val.txt')

    image_paths = [x[0] for x in train_data + val_data]
    labels = [x[1] for x in train_data + val_data]
    
    # 保存类别索引（基于完整数据集）
    save_class_indices(image_paths, labels)

    # 统计数据集分布
    train_labels = [x[1] for x in train_data]
    val_labels = [x[1] for x in val_data]

    print("\n训练集统计:")
    print("总样本数:", len(train_data))
    print("类别分布:", dict(sorted(Counter(train_labels).items())))

    print("\n验证集统计:")
    print("总样本数:", len(val_data))
    print("类别分布:", dict(sorted(Counter(val_labels).items())))

    # 定义数据增强流程
    train_transform = transforms.Compose([
        transforms.Resize(args.size, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        Cutout(),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(args.size),
        transforms.ToTensor(),
    ])

    # 创建数据集
    train_dataset = MyTrainDataset(train_data, transform=train_transform, augment4x=args.augment4x == 'aug4x')
    val_dataset = MyDataset(val_data, transform=val_transform)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8
    )

    # 初始化模型
    model, model_name = load_model(args)

    # 训练配置（保持原有优化策略）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-6, weight_decay=1e-3)
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-9)
    
    # 训练循环
    train_losses, val_losses = [], []
    best_acc = 0.0
    best_ckpt_path = None
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train(model, train_loader, optimizer, args, criterion, epoch, train_losses)
        
        # 验证并保存最佳模型
        current_acc, val_loss, ckpt_path = val(
            model, val_loader, args, criterion, epoch, val_losses, 
            ACC=best_acc, best_loss=float('inf'), 
            best_ckpt_path=None, lr=args.lr, model_name=model_name
        )
        
        if current_acc > best_acc:
            best_acc = current_acc
            best_ckpt_path = ckpt_path
        
        cosine_schedule.step()

    # 最终预测和绘图
    print('******************* FINAL PREDICTION *******************')
    Predictor(model, val_loader, best_ckpt_path, args.device, fold=0)
    plot_losses(train_losses, val_losses, fold=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
