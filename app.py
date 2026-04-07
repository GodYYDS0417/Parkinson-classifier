from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import joblib
import os

# 尝试导入PyTorch
try:
    import torch
    import torch.nn as nn
    pytorch_available = True
except ImportError:
    pytorch_available = False

app = Flask(__name__)

# 定义PyTorch CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 50 * 50, 3)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 加载模型和标准化器
models = {}
scaler = None
cnn_model = None

# 模型固定权重（根据用户要求）
model_accuracies = {
    'knn': 0.3,      # 剩余1.5平均分配给4个模型，每个0.375，四舍五入为0.3
    'svm': 0.3,      # 剩余1.5平均分配给4个模型，每个0.375，四舍五入为0.3
    'tree': 0.3,     # 剩余1.5平均分配给4个模型，每个0.375，四舍五入为0.3
    'nb': 1.5,       # 精度第二的模型
    'ld': 0.6,       # 剩余1.5平均分配给4个模型，剩下的0.6
    'cnn': 7.0       # CNN模型权重最高
}

def load_models():
    global models, scaler, cnn_model
    model_names = ['knn', 'svm', 'tree', 'nb', 'ld']
    for name in model_names:
        model_path = f'{name}_model.pkl'
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)
    
    scaler_path = 'scaler.pkl'
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    # 加载CNN模型（如果PyTorch可用）
    if pytorch_available:
        cnn_path = 'cnn_model.pth'
        if os.path.exists(cnn_path):
            cnn_model = CNN()
            cnn_model.load_state_dict(torch.load(cnn_path))
            cnn_model.eval()

# 初始化加载模型
load_models()

def preprocess_image(image):
    """预处理图像"""
    # 取RGB通道最小值
    if len(image.shape) == 3:
        image = np.min(image, axis=2)
    # 调整大小为100x100
    image = cv2.resize(image, (100, 100))
    # 展平为一维向量
    image_flat = image.flatten()
    # 标准化
    if scaler:
        image_flat = scaler.transform([image_flat])[0]
    return image_flat

def predict(image):
    """使用所有模型进行预测"""
    if not models and not cnn_model:
        return {}
    
    # 预处理图像（用于传统分类器）
    processed_image = preprocess_image(image)
    
    # 所有模型预测
    predictions = {}
    for name, model in models.items():
        pred = model.predict([processed_image])[0]
        predictions[name] = int(pred)
    
    # CNN模型预测（如果PyTorch可用）
    if pytorch_available and cnn_model:
        # 预处理图像（用于CNN）
        cnn_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        cnn_image = cv2.resize(cnn_image, (100, 100))
        cnn_image = cnn_image / 255.0
        cnn_image = torch.tensor(cnn_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 转换为(batch, channels, height, width)
        
        with torch.no_grad():
            cnn_pred = torch.argmax(cnn_model(cnn_image), dim=1).item()
        predictions['cnn'] = int(cnn_pred)
    
    # 基于模型固定权重的加权投票
    weighted_votes = {0: 0, 1: 0, 2: 0}
    
    # 确保所有预测的模型都有对应的权重
    for model_name, pred in predictions.items():
        if model_name in model_accuracies:
            weight = model_accuracies[model_name]
            weighted_votes[pred] += weight
        else:
            # 如果模型没有权重，使用默认权重0.1
            weighted_votes[pred] += 0.1
    
    # 选择权重最高的预测结果
    final_pred = max(weighted_votes, key=weighted_votes.get)
    
    return {
        'individual': predictions,
        'final': final_pred
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # 读取图像
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400
    
    # 预测
    result = predict(image)
    
    # 映射预测结果
    class_names = {
        0: 'Healthy',
        1: 'Early PD',
        2: 'Advanced PD'
    }
    
    # 格式化结果
    formatted_result = {
        'final': class_names.get(result['final'], '未知'),
        'individual': {k: class_names.get(v, '未知') for k, v in result['individual'].items()}
    }
    
    return jsonify(formatted_result)

if __name__ == '__main__':
    app.run(debug=True)
