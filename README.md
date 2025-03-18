# VGG-19 and Basic CNN Image Classification Models

This project implements two image classification models:
1. A **VGG-19 based model** (`moderate_model_vgg_19.ipynb`) for deep feature extraction and classification.
2. A **Basic CNN model** (`basic_model3.ipynb`) with data augmentation for image classification.

## Features
- Uses **Convolutional Neural Networks (CNNs)** for image classification.
- **Data Preprocessing:** Image augmentation, class balancing (SMOTE), and label encoding.
- **Training and Evaluation:** Model training, performance metrics, and visualization.
- **Optimizers:** Uses RMSprop and Adam for training.

## Installation
Ensure you have the following dependencies installed:

```bash
pip install tensorflow numpy pandas scikit-learn imbalanced-learn tqdm matplotlib
```

## Dataset
Both models work with image data referenced in CSV files (e.g., `MURA-v1.1/valid_image_paths_with_labels.csv`). Ensure the dataset is available before running the notebooks.

## Usage
Run the Jupyter Notebooks:

```bash
jupyter notebook moderate_model_vgg_19.ipynb
jupyter notebook basic_model3.ipynb
```

### Steps:
1. Load and preprocess image data.
2. Define CNN architectures with convolutional, pooling, and dropout layers.
3. Train models with training and validation datasets.
4. Evaluate performance using classification metrics.
5. Visualize results using matplotlib.

## Model Architectures

### VGG-19 Based Model (`moderate_model_vgg_19.ipynb`)
- **Conv2D Layers**: Extract spatial features from images.
- **MaxPooling2D**: Downsample feature maps.
- **Flatten Layer**: Converts feature maps into a vector.
- **Dense Layers**: Fully connected layers for classification.
- **Dropout Layers**: Prevent overfitting.
- **Optimizer**: RMSprop.
- **Loss Function**: Binary Crossentropy.

### Basic CNN Model (`basic_model3.ipynb`)
- **ImageDataGenerator**: Data augmentation (flipping, rotation, etc.).
- **Conv2D Layers**: Extract spatial features.
- **MaxPooling2D**: Downsampling.
- **Dense Layers**: Fully connected layers for classification.
- **Softmax Activation**: Multi-class classification.
- **Optimizers**: Adam and RMSprop.
- **Loss Function**: Binary Crossentropy.

## Evaluation Metrics
- **Accuracy**
- **Confusion Matrix**
- **Classification Report**
- **F-beta Score (Beta=2)**
- **Cohen’s Kappa Score**

## Results
After training, both models generate predictions on test data. Results can be visualized using matplotlib to analyze accuracy and loss trends.

## Contributors
- **Bhavya Gupta**  
- Feel free to contribute by improving model performance or optimizing the architecture.

## License
This project is licensed under the MIT License.

