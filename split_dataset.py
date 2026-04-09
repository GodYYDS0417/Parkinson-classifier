import os
import cv2
import numpy as np

# 原始数据集路径
original_dataset_path = 'c:\\Users\\28130\\Desktop\\Parkinson-classifier-master\\PD_datasets'
# 保存分割后的数据集路径
split_dataset_path = 'c:\\Users\\28130\\Desktop\\Parkinson-classifier-master\\split_PD_datasets'

# 创建保存目录
os.makedirs(os.path.join(split_dataset_path, '0'), exist_ok=True)  # 健康
os.makedirs(os.path.join(split_dataset_path, '1'), exist_ok=True)  # 早期
os.makedirs(os.path.join(split_dataset_path, '2'), exist_ok=True)  # 中期
os.makedirs(os.path.join(split_dataset_path, '3'), exist_ok=True)  # 晚期

def split_and_save_images():
    """分割图像并保存"""
    total_samples = 0
    
    # 遍历原始数据集
    for class_folder in ['0', '1', '2']:
        class_path = os.path.join(original_dataset_path, class_folder)
        if not os.path.exists(class_path):
            continue
        
        # 遍历图像文件
        for filename in os.listdir(class_path):
            if not (filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg')):
                continue
            
            # 读取图像
            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # 分割图像
            height, width, _ = image.shape
            left_half = image[:, :width//2]  # 左侧（健康）
            right_half = image[:, width//2:]  # 右侧（帕金森）
            
            # 保存左侧（健康）
            healthy_filename = f"{class_folder}_{filename}"
            healthy_path = os.path.join(split_dataset_path, '0', healthy_filename)
            cv2.imwrite(healthy_path, left_half)
            total_samples += 1
            
            # 保存右侧（帕金森）
            pd_label = str(int(class_folder) + 1)  # 0→1（早期）, 1→2（中期）, 2→3（晚期）
            pd_filename = f"{class_folder}_{filename}"
            pd_path = os.path.join(split_dataset_path, pd_label, pd_filename)
            cv2.imwrite(pd_path, right_half)
            total_samples += 1
            
            # 数据增强：水平翻转右侧图像
            flipped_right = cv2.flip(right_half, 1)
            flipped_filename = f"{class_folder}_{filename.split('.')[0]}_flipped.{filename.split('.')[-1]}"
            flipped_path = os.path.join(split_dataset_path, pd_label, flipped_filename)
            cv2.imwrite(flipped_path, flipped_right)
            total_samples += 1
    
    print(f"分割完成，共保存 {total_samples} 个样本")

if __name__ == "__main__":
    split_and_save_images()
