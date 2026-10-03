# Skin Cancer Detection Using Deep Learning

A deep learning-based image classification system for detecting **benign and malignant skin lesions** using a custom Convolutional Neural Network (CNN).

The project is designed as an end-to-end AI/ML application covering dataset preprocessing, model training, evaluation, prediction, and a Streamlit web interface.

---

## Project Overview

Skin cancer detection is an important application of computer vision in healthcare. This project uses a CNN to classify dermoscopic skin lesion images into two categories:

- **Benign**
- **Malignant**

The model takes a skin lesion image as input, preprocesses it to a fixed size, and predicts the probability of the lesion being malignant.

> **Medical Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic system and should not be used for clinical decisions.

---

## Features

- Binary skin lesion classification
- Custom CNN architecture using PyTorch
- Image preprocessing and resizing
- Class-imbalance handling using weighted binary cross-entropy
- Reproducible train/validation split
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
  - ROC-AUC
- Command-line image prediction
- Streamlit web application
- GPU acceleration when CUDA is available
- Modular and easy-to-understand project structure

---

## Model Architecture

```text
Input Image (224 x 224 x 3)
        |
        v
Conv2D (3 -> 16)
        |
        v
ReLU
        |
        v
Max Pooling
        |
        v
Conv2D (16 -> 32)
        |
        v
ReLU
        |
        v
Max Pooling
        |
        v
Feature Map (32 x 56 x 56)
        |
        v
Flatten
        |
        v
Fully Connected (32 x 56 x 56 -> 128)
        |
        v
ReLU
        |
        v
Fully Connected (128 -> 1)
        |
        v
Output Logit
        |
        v
Sigmoid
        |
        v
Malignancy Probability
        |
        v
Benign / Malignant
```

---

## Dataset

The project uses skin lesion image data with binary labels:

| Class | Label |
|---|---:|
| Benign | 0 |
| Malignant | 1 |

The dataset is **not included in this repository** because of its size and dataset licensing/privacy considerations.

The expected processed label file is:

```text
data/processed_labels.csv
```

Expected columns:

```text
image_name
target
```

Images should be available in:

```text
data/resized_images/
```

---

## Handling Class Imbalance

Skin lesion datasets can contain significantly more benign samples than malignant samples.

To reduce the effect of this imbalance, the training process uses a weighted `BCEWithLogitsLoss`.

The positive-class weight is calculated as:

```text
positive_weight = number_of_benign_samples / number_of_malignant_samples
```

This gives greater importance to malignant samples during training.

---

## Project Structure

```text
Skin_cancer_detection/
|
├── app.py
├── dataset.py
├── evaluate.py
├── model.py
├── predict.py
├── train.py
|
├── data/
│   ├── processed_labels.csv
│   └── resized_images/
|
├── models/
│   └── skin_cancer_model.pt
|
├── scripts/
│   ├── balance_dataset.py
│   ├── check_labels.py
│   ├── clean_labels.py
│   ├── filter_valid_labels.py
│   ├── preprocess.py
│   ├── resize_images.py
│   └── test_prediction.py
|
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SupR2344/Skin_cancer_detection.git
cd Skin_cancer_detection
```

### 2. Create a virtual environment

Using Python:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

Place the processed dataset in the following structure:

```text
data/
├── processed_labels.csv
└── resized_images/
    ├── image_001.jpg
    ├── image_002.jpg
    └── ...
```

Each image name in `processed_labels.csv` should correspond to an image inside `data/resized_images/`.

---

## Training

If you want to train the CNN from scratch:

```bash
python train.py
```

The trained model is saved as:

```text
models/skin_cancer_model.pt
```

The training script automatically uses a CUDA-enabled GPU when available.

---

## Model Evaluation

Run:

```bash
python evaluate.py
```

The evaluation process generates classification metrics and evaluates the model on the validation dataset.

The evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

---

## Image Prediction

To classify a single image:

```bash
python predict.py --image path/to/image.jpg
```

The prediction returns:

- Predicted class
- Malignant probability
- Benign probability

Example:

```text
Prediction: Malignant
Malignant Probability: 0.87
Benign Probability: 0.13
```

---

## Streamlit Application

The project also provides a web-based interface.

Run:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

The application allows users to:

1. Upload a skin lesion image.
2. Preview the image.
3. Run the trained CNN.
4. View the predicted class.
5. View the prediction probability.

---

## GPU Support

The project supports NVIDIA CUDA through PyTorch.

Check whether CUDA is available:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is available, training and inference can use the GPU automatically.

---

## Technologies Used

- **Python**
- **PyTorch**
- **Torchvision**
- **Pandas**
- **NumPy**
- **Pillow**
- **Scikit-learn**
- **Matplotlib**
- **Streamlit**
- **CUDA**

---

## Machine Learning Workflow

```text
Dataset
   |
   v
Data Cleaning
   |
   v
Label Processing
   |
   v
Image Resizing
   |
   v
Train / Validation Split
   |
   v
CNN Training
   |
   v
Class-weighted Loss
   |
   v
Trained Model
   |
   +---------> Evaluation
   |
   +---------> Single Image Prediction
   |
   +---------> Streamlit Application
```

---

## Limitations

This project has several limitations:

- It uses a relatively simple custom CNN architecture.
- Performance depends heavily on dataset quality.
- Class imbalance can affect prediction performance.
- The model may not generalize well to images from different datasets or imaging devices.
- It has not been clinically validated.
- Predictions should not be interpreted as medical diagnoses.

---

## Future Improvements

Possible improvements include:

- Transfer learning using ResNet, EfficientNet, or ConvNeXt
- Data augmentation
- Better class-imbalance techniques
- Hyperparameter optimization
- Explainable AI using Grad-CAM
- Model calibration
- Cross-validation
- Improved Streamlit UI
- Model deployment using Docker
- Cloud deployment
- Experiment tracking
- Integration with a larger and more diverse dataset

---

## Author

**Supriyo Rana**

AI/ML Engineering Student

- GitHub: https://github.com/SupR2344
- LinkedIn: https://www.linkedin.com/in/supriyo-rana-337ab4351/

---

## License

This project is licensed under the **MIT License**.
