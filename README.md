# Parkinson's Disease Image Classification System

This project implements a Parkinson's disease image classification system using multiple machine learning models, including KNN, SVM, Decision Tree, Naive Bayes, Linear Discriminant Analysis, and CNN.

## Project Structure

```
├── code_PD/               # MATLAB code for data preprocessing and model training
│   ├── Accuracy_Datapreprocess.xlsx
│   ├── Data_preprocess.m
│   ├── Loss_Datapreprocess.xlsx
│   ├── customreader.m
│   ├── f1_score.m
│   ├── plot_all_figure.m
│   ├── trainClassifier_KNN.m
│   ├── trainClassifier_LD.m
│   ├── trainClassifier_NB.m
│   ├── trainClassifier_SVM.m
│   ├── trainClassifier_Tree.m
│   └── writedata.m
├── templates/             # Flask web template
│   └── index.html         # Main web page
├── app.py                 # Flask application
├── train_model.py         # Python model training script
├── requirements.txt       # Python dependencies
└── .gitignore             # Git ignore file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/GodYYDS0417/Parkinson-classifier.git
   cd Parkinson-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the models

Run the training script to train all models:

```bash
python train_model.py
```

This will generate the following model files:
- `knn_model.pkl`
- `svm_model.pkl`
- `tree_model.pkl`
- `nb_model.pkl`
- `ld_model.pkl`
- `cnn_model.pth`
- `scaler.pkl` (for data normalization)

### 2. Run the web application

Start the Flask web server:

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000` to use the web interface.

## Model Weights

You can download the pre-trained model weights from the following link:

- **Link**: [https://pan.baidu.com/s/1_lYi_c3V4hv9m-wUQalADw?pwd=5us6](https://pan.baidu.com/s/1_lYi_c3V4hv9m-wUQalADw?pwd=5us6)
- **Password**: 5us6

Download the checkpoints and place them in the project root directory.

## Model Architecture

### Traditional Machine Learning Models
- **KNN**: K-Nearest Neighbors
- **SVM**: Support Vector Machine
- **Decision Tree**: CART Decision Tree
- **Naive Bayes**: Gaussian Naive Bayes
- **LDA**: Linear Discriminant Analysis

### CNN Model
The CNN model consists of:
- 1 convolutional layer with 32 filters
- Batch normalization
- ReLU activation
- Max pooling
- Fully connected layer with 3 output classes

## Classification Results

The system classifies images into three categories:
- **Healthy**: No Parkinson's disease
- **Early PD**: Early stage Parkinson's disease
- **Advanced PD**: Advanced stage Parkinson's disease

## Weighted Voting

The final prediction uses a weighted voting system based on model performance:
- CNN: 7.0
- Naive Bayes: 1.5
- KNN: 0.3
- SVM: 0.3
- Decision Tree: 0.3
- LDA: 0.6

## License

This project is for research purposes only.
