import os
import cv2
import numpy as np
import joblib
import torch
import torch.nn as nn

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 输入是100x100x3的RGB图像
        # 卷积层：3x3卷积核，32个输出通道，same padding
        self.conv = nn.Conv2d(3, 32, 3, padding=1)  # 3x3卷积核，padding=1保持输出大小
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 100 * 100, 4)  # 4个类别：健康、早期、中期、晚期
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def preprocess_image(image):
    """预处理图像，保持比例，避免拉伸变形"""
    # 保持RGB通道
    
    # 获取图像尺寸
    height, width, channels = image.shape
    
    # 计算最大边
    max_dim = max(height, width)
    
    # 创建正方形画布，边缘补0
    square_image = np.zeros((max_dim, max_dim, channels), dtype=np.uint8)
    
    # 计算居中位置
    h_start = (max_dim - height) // 2
    w_start = (max_dim - width) // 2
    
    # 将原始图像复制到画布中心
    square_image[h_start:h_start+height, w_start:w_start+width] = image
    
    # 等比例缩放到100x100
    image = cv2.resize(square_image, (100, 100))
    
    # 展平为一维向量
    image = image.flatten()
    
    # 标准化
    if scaler:
        image = scaler.transform([image])[0]
    
    return image

def test_model():
    """测试模型"""
    global scaler, cnn_model, models
    
    # 加载模型
    checkpoints_dir = 'checkpoints'
    model_names = ['knn', 'svm', 'tree', 'nb', 'ld']
    models = {}
    
    for name in model_names:
        model_path = f'{checkpoints_dir}/{name}_model.pkl'
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)
            print(f"Loaded {name} model")
    
    # 加载标准化器
    scaler_path = f'{checkpoints_dir}/scaler.pkl'
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("Loaded scaler")
    
    # 加载CNN模型
    cnn_path = f'{checkpoints_dir}/cnn_model.pth'
    if os.path.exists(cnn_path):
        cnn_model = CNN()
        cnn_model.load_state_dict(torch.load(cnn_path))
        cnn_model.eval()
        print("Loaded CNN model")
    
    # 测试样本路径
    test_data_path = '../split_PD_datasets'
    
    # 类别标签
    class_names = {0: 'Healthy', 1: 'Early', 2: 'Medium', 3: 'Late'}
    
    # 测试每个类别的样本
    for class_label in os.listdir(test_data_path):
        class_path = os.path.join(test_data_path, class_label)
        if not os.path.isdir(class_path):
            continue
        
        print(f"\nTesting class: {class_names[int(class_label)]}")
        
        # 测试前5个样本
        count = 0
        for filename in os.listdir(class_path):
            if count >= 5:
                break
            
            if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
                image_path = os.path.join(class_path, filename)
                print(f"\nTesting image: {filename}")
                
                # 读取图像
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to read image: {filename}")
                    continue
                
                # 预处理图像
                processed = preprocess_image(image)
                
                # 传统模型预测
                print("Traditional models predictions:")
                for name, model in models.items():
                    pred = model.predict([processed])[0]
                    print(f"{name}: {class_names[pred]}")
                
                # CNN模型预测
                if cnn_model:
                    # 预处理用于CNN
                    # 保持RGB通道
                    height, width, channels = image.shape
                    max_dim = max(height, width)
                    square_image = np.zeros((max_dim, max_dim, channels), dtype=np.uint8)
                    h_start = (max_dim - height) // 2
                    w_start = (max_dim - width) // 2
                    square_image[h_start:h_start+height, w_start:w_start+width] = image
                    cnn_image = cv2.resize(square_image, (100, 100))
                    cnn_image = cnn_image / 255.0
                    cnn_image = torch.tensor(cnn_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # 转换为(batch, channels, height, width)
                    
                    with torch.no_grad():
                        cnn_pred = torch.argmax(cnn_model(cnn_image), dim=1).item()
                    print(f"CNN: {class_names[cnn_pred]}")
                
                count += 1

if __name__ == '__main__':
    test_model()
