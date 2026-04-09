from flask import Flask, render_template, request, jsonify, send_from_directory
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
app.config['UPLOAD_FOLDER'] = '.'
app.config['STATIC_FOLDER'] = '.'

# 定义PyTorch CNN模型
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

# 加载模型和标准化器
models = {}
scaler = None
cnn_model = None

# 模型固定权重（根据用户要求）
model_accuracies = {
    'knn': 0.1,      # 降低其他模型权重
    'svm': 0.1,      # 降低其他模型权重
    'tree': 0.1,     # 降低其他模型权重
    'nb': 0.5,       # 降低其他模型权重
    'ld': 0.2,       # 降低其他模型权重
    'cnn': 9.0       # 提高CNN模型权重
}

def load_models():
    global models, scaler, cnn_model
    model_names = ['knn', 'svm', 'tree', 'nb', 'ld']
    checkpoints_dir = 'checkpoints'
    for name in model_names:
        model_path = f'{checkpoints_dir}/{name}_model.pkl'
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)
    
    scaler_path = f'{checkpoints_dir}/scaler.pkl'
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    # 加载CNN模型（如果PyTorch可用）
    if pytorch_available:
        cnn_path = f'{checkpoints_dir}/cnn_model.pth'
        if os.path.exists(cnn_path):
            cnn_model = CNN()
            cnn_model.load_state_dict(torch.load(cnn_path))
            cnn_model.eval()

# 初始化加载模型
load_models()
print(f"Models loaded: {models}")
print(f"CNN model: {cnn_model}")
print(f"PyTorch available: {pytorch_available}")

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

def predict(image):
    """使用所有模型进行预测"""
    if not models and not cnn_model:
        return {}
    
    # 提取左侧（健康）和右侧（帕金森）
    height, width, _ = image.shape
    print(f"Image shape: {height}x{width}")
    left_half = image[:, :width//2]
    right_half = image[:, width//2:]
    print(f"Left half shape: {left_half.shape}")
    print(f"Right half shape: {right_half.shape}")
    
    # 预处理左侧（健康）
    left_processed = preprocess_image(left_half)
    # 预处理右侧（帕金森）
    right_processed = preprocess_image(right_half)
    print(f"Right processed shape: {right_processed.shape}")
    
    # 所有模型预测
    predictions = {}
    
    # 传统模型预测右侧（帕金森）
    for name, model in models.items():
        pred = model.predict([right_processed])[0]
        predictions[name] = int(pred)
    
    # CNN模型预测右侧（帕金森）（如果PyTorch可用）
    if pytorch_available and cnn_model:
        # 预处理右侧图像（用于CNN）
        # 保持RGB通道
        # 保持比例，避免拉伸变形
        height, width, channels = right_half.shape
        max_dim = max(height, width)
        square_image = np.zeros((max_dim, max_dim, channels), dtype=np.uint8)
        h_start = (max_dim - height) // 2
        w_start = (max_dim - width) // 2
        square_image[h_start:h_start+height, w_start:w_start+width] = right_half
        cnn_image = cv2.resize(square_image, (100, 100))
        
        cnn_image = cnn_image / 255.0
        cnn_image = torch.tensor(cnn_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # 转换为(batch, channels, height, width)
        
        with torch.no_grad():
            cnn_pred = torch.argmax(cnn_model(cnn_image), dim=1).item()
        predictions['cnn'] = int(cnn_pred)
        print(f"CNN prediction: {cnn_pred}")
    else:
        print(f"PyTorch available: {pytorch_available}, CNN model: {cnn_model}")
    
    print(f"Predictions: {predictions}")
    
    # 确保至少有一个预测结果
    if not predictions:
        return {}
    
    # 强制使用CNN模型的预测结果
    if 'cnn' in predictions:
        final_pred = predictions['cnn']
        print(f"Using CNN prediction: {final_pred}")
    else:
        # 如果没有CNN预测，使用加权投票
        weighted_votes = {0: 0, 1: 0, 2: 0, 3: 0}
        
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
        print(f"Using weighted voting: {final_pred}")
    
    return {
        'individual': predictions,
        'final': final_pred
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

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
    
    # 检查是否有模型可用
    if not result:
        return jsonify({'error': 'No models available. Please train the models first.'}), 500
    
    # 映射预测结果
    class_names = {
        0: 'Healthy',
        1: 'Early',
        2: 'Medium',
        3: 'Late'
    }
    
    # 格式化结果
    formatted_result = {
        'final': class_names.get(result['final'], '未知'),
        'individual': {k: class_names.get(v, '未知') for k, v in result['individual'].items()}
    }
    
    return jsonify(formatted_result)

if __name__ == '__main__':
    app.run(debug=True)
