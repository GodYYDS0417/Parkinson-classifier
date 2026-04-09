import os
import numpy as np
import cv2
from sklearn.model_selection import cross_val_score, KFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 简化数据增强函数
def augment_image(image):
    """简化的数据增强函数"""
    augmented = []
    
    # 原始图像
    augmented.append(image)
    
    # 水平翻转
    flipped = cv2.flip(image, 1)
    augmented.append(flipped)
    
    return augmented

from sklearn.preprocessing import StandardScaler
import joblib

# 尝试导入PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    pytorch_available = True
except ImportError:
    pytorch_available = False

def customreader(filename):
    """读取并预处理图像，保持比例，避免拉伸变形"""
    data = cv2.imread(filename)
    if data is None:
        return None
    
    # 保持比例，避免拉伸变形
    height, width, channels = data.shape
    max_dim = max(height, width)
    square_image = np.zeros((max_dim, max_dim, channels), dtype=np.uint8)
    h_start = (max_dim - height) // 2
    w_start = (max_dim - width) // 2
    square_image[h_start:h_start+height, w_start:w_start+width] = data
    data = cv2.resize(square_image, (100, 100))
    

    return data

def load_data(dataset_path):
    """加载数据集"""
    data = []
    data_cnn = []
    labels = []
    
    # 遍历每个类别文件夹
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            # 类别标签：健康=0，早期=1，中期=2，晚期=3
            class_label = int(label)
            
            # 遍历每个图像文件
            for filename in os.listdir(label_path):
                if filename.endswith('.tif') or filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
                    img_path = os.path.join(label_path, filename)
                    img = customreader(img_path)
                    if img is not None:
                        # 添加样本
                        # 对于传统模型，展平为一维向量
                        img_flat = img.flatten()
                        data.append(img_flat)
                        # 对于CNN模型，保持3通道
                        data_cnn.append(img)
                        labels.append(class_label)

    
    return np.array(data), np.array(data_cnn), np.array(labels)

def train_models(X, X_cnn, y):
    """训练多种分类器"""
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 创建checkpoints目录
    os.makedirs('checkpoints', exist_ok=True)
    
    # 保存标准化器
    joblib.dump(scaler, 'checkpoints/scaler.pkl')

    
    # 定义分类器
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=1, metric='euclidean'),
        'SVM': SVC(kernel='rbf', C=1, gamma=1/87, probability=True),
        'Tree': DecisionTreeClassifier(max_depth=None, max_leaf_nodes=None),
        'NB': GaussianNB(),
        'LD': LinearDiscriminantAnalysis()
    }
    
    # 训练并评估每个分类器
    results = {}
    for name, clf in classifiers.items():
        # 5折交叉验证
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_scaled, y, cv=kfold, scoring='accuracy')
        
        # 训练完整模型
        clf.fit(X_scaled, y)
        
        # 保存模型
        joblib.dump(clf, f'checkpoints/{name.lower()}_model.pkl')

        
        results[name] = {
            'accuracy': np.mean(scores),
            'std': np.std(scores)
        }
        
        print(f'{name}: 准确率 = {np.mean(scores):.4f} ± {np.std(scores):.4f}')
    
    # 训练CNN模型（如果PyTorch可用）
    if pytorch_available:
        print('训练CNN模型中...')
        
        # 定义PyTorch模型（根据MATLAB代码逻辑，使用更小的卷积核以提高训练速度）
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
        
        # 准备PyTorch数据
        X_cnn = X_cnn / 255.0  # 归一化
        X_cnn = torch.tensor(X_cnn, dtype=torch.float32).permute(0, 3, 1, 2)  # 转换为(batch, channels, height, width)
        y_cnn = torch.tensor(y, dtype=torch.long)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(X_cnn, y_cnn)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # 初始化模型、损失函数和优化器
        model = CNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.0001)  # 使用SGDM优化器，与MATLAB一致
        
        # 训练模型
        epochs = 50  # 训练50轮

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in dataloader:
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                # 统计损失和准确率
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = correct / total
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        
        # 保存模型
        torch.save(model.state_dict(), 'checkpoints/cnn_model.pth')

        
        # 评估模型
        model.eval()
        with torch.no_grad():
            outputs = model(X_cnn)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_cnn).sum().item() / len(y_cnn)
        
        results['CNN'] = {
            'accuracy': accuracy,
            'std': 0.0
        }
        print(f'CNN: 准确率 = {accuracy:.4f}')
    else:
        print('PyTorch不可用，跳过CNN模型训练')
    
    return results

def main():
    # 数据集路径
    dataset_path = '../split_PD_datasets'

    
    # 加载数据
    print('加载数据中...')
    X, X_cnn, y = load_data(dataset_path)
    print(f'数据加载完成，共 {len(X)} 个样本')
    
    # 训练模型
    print('训练模型中...')
    results = train_models(X, X_cnn, y)
    
    # 打印结果
    print('\n训练结果:')
    for name, result in results.items():
        print(f'{name}: 准确率 = {result["accuracy"]:.4f} ± {result["std"]:.4f}')

if __name__ == '__main__':
    main()
