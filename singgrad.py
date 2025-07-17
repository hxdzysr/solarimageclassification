import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import GradCAM, show_cam_on_image
import torchvision.models as models

class ReshapeTransform:
    def __init__(self, model):
        input_size = (224,224)
        patch_size = model.conv_proj.kernel_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result

def vit_gradcam(path, cate, vitweights, device):
    model = models.vit_b_16(weights=None)
    model.heads = torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=11))
    weights = vitweights
    checkpoint = torch.load(weights, map_location=device)
    # 正确提取模型参数
    if 'state_dict' in checkpoint:  # 如果检查点包含完整训练状态
        state_dict = checkpoint['state_dict']
    else:  # 如果检查点直接保存的是模型参数
        state_dict = checkpoint
    # 处理可能的DataParallel前缀（如果模型在GPU上训练时使用了并行）
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    target_layers = [model.encoder.layers.encoder_layer_10]
    data_transform = transforms.Compose([transforms.ToTensor(),
                                        ])
    # load image
    img_path = path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = transforms.Resize((224, 224))(img)
    #img = transforms.RandomVerticalFlip(p=1)(img)
    #img = transforms.RandomHorizontalFlip(p=1)(img)
    img = np.array(img, dtype=np.uint8)
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=False,
                  reshape_transform=ReshapeTransform(model))
    target_category = cate

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))  # 每行2个子图
    axes = axes.flatten()
    img_ax = axes[0]
    img_ax.imshow(visualization)
    ori_ax = axes[1]
    ori_ax.imshow(img_tensor.permute(1, 2, 0).numpy())
    plt.tight_layout()
    plt.show()

def cnn_gradcam(path, cate, cnnweights, device):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(in_features=2048, out_features=11, bias=True)
    weights = cnnweights
    checkpoint = torch.load(weights, map_location=device)
    # 正确提取模型参数
    if 'state_dict' in checkpoint:  # 如果检查点包含完整训练状态
        state_dict = checkpoint['state_dict']
    else:  # 如果检查点直接保存的是模型参数
        state_dict = checkpoint
    # 处理可能的DataParallel前缀（如果模型在GPU上训练时使用了并行）
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    target_layers = [model.layer4]
    data_transform = transforms.Compose([transforms.ToTensor()])

    img_path = path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = transforms.Resize((224, 224))(img)
    #img = transforms.RandomVerticalFlip(p=1)(img)
    #img = transforms.RandomHorizontalFlip(p=1)(img)
    img = np.array(img, dtype=np.uint8)
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=False,)
    target_category = cate

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))  # 每行2个子图
    axes = axes.flatten()
    img_ax = axes[0]
    img_ax.imshow(visualization)
    ori_ax = axes[1]
    ori_ax.imshow(img_tensor.permute(1, 2, 0).numpy())
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
     path = '/Users/cosmicdawn/Downloads/ViT/data/01-photo-sunspot/Tio_20190412_010708_1B.jpeg'
     cate = 0
     vitweights = "vit_acc0.94_fold0_epoch52_lr0.0001_loss0.7137122316793962.pth"
     cnnweights = "res_acc0.92_epoch68_lr0.0001_loss0.8197019303717265.pth"
     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     vit_gradcam(path, cate, vitweights, device)
     cnn_gradcam(path, cate, cnnweights, device)
