import os
import random

def generate_stratified_split(data_dir, output_train, output_val, output_test):
    """
    实现分层8:1:1分割，仅保留文件夹类别标签
    数据结构：文件路径,类别标签
    """
    class_data = {}  # 按类别存储数据
    folder_names = sorted([f for f in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, f))])
    
    # 创建类别标签映射字典
    folder_label_map = {folder: idx for idx, folder in enumerate(folder_names)}
    print("类别标签映射：", folder_label_map)

    # 遍历收集数据
    for root, dirs, files in os.walk(data_dir):
        folder_name = os.path.basename(root)
        if folder_name not in folder_names:
            continue
            
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                # 处理路径并生成标签
                full_path = os.path.join(root, file).replace("\\", "/")
                label = folder_label_map[folder_name]
                
                # 按类别存储
                if folder_name not in class_data:
                    class_data[folder_name] = []
                class_data[folder_name].append((full_path, label))

    # 分割数据集
    train, val, test = [], [], []
    for class_name, items in class_data.items():
        random.shuffle(items)
        n = len(items)
        
        # 计算分割点
        split1 = int(0.8 * n)
        split2 = int(0.9 * n)
        
        train += items[:split1]
        val += items[split1:split2]
        test += items[split2:]

    # 最终打乱（保持各类别混合后的随机性）
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    # 写入文件函数
    def save_dataset(data, filename):
        with open(filename, 'w') as f:
            for path, label in data:
                f.write(f"{path},{label}\n")

    # 保存数据集
    save_dataset(train, output_train)
    save_dataset(val, output_val)
    save_dataset(test, output_test)

# 使用示例
generate_stratified_split(
    data_dir='./newdata',
    output_train='multrain.txt',
    output_val='mulval.txt',
    output_test='multest.txt'
)