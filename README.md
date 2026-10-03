# Skin Cancer Detection

A binary image classifier built with PyTorch that predicts whether an uploaded
skin-lesion image is **benign** or **malignant**. A Streamlit app is provided
for running predictions with the included trained model.

> **Medical disclaimer:** This project is for educational and research use only.
> It is not a medical device and must not be used to diagnose, treat, or rule
> out skin cancer. Consult a qualified healthcare professional for medical
> advice.

## Project structure

```text
.
├── data/
│   ├── *.csv                  # Labels and metadata
│   └── images/malignant/      # Dataset attribution and license information
├── models/
│   └── skin_cancer_model.pt   # Trained PyTorch state dictionary
├── scripts/
│   ├── app.py                 # Streamlit prediction interface
│   ├── model.py               # CNN architecture
│   ├── predict.py             # Command-line prediction helper
│   ├── train.py               # PyTorch training script
│   └── ...                    # Dataset preparation and evaluation scripts
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

Image files are not included in the repository. Dataset labels and the
attribution/license files are included; see `data/images/malignant/` before
using any associated image data. The pretrained weights are included in
`models/skin_cancer_model.pt`.

## Setup

Use Python 3.10 or newer, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with
`source .venv/bin/activate` instead.

## Run the app

From the repository root:

```powershell
streamlit run scripts/app.py
```

Open the local URL printed by Streamlit and upload a JPG or PNG image. The app
resizes images to 224 × 224 pixels and displays the model's binary prediction
and sigmoid output.

## Train and evaluate

The PyTorch training and evaluation scripts expect a dataset image tree rooted
at `data/resized_images/`. Each row in `data/balanced_labels.csv` has an
`image_name` path relative to that directory and a binary `target` (`0` for
benign, `1` for malignant). Place the corresponding images at those paths
before running:

```powershell
python scripts/train.py
python scripts/evaluate.py
```

Training writes the resulting state dictionary to
`models/skin_cancer_model.pt`, replacing the included weights. Evaluation
reports classification metrics and plots a confusion matrix and ROC curve.

## Data and model notes

- The model is a small convolutional neural network defined in
  `scripts/model.py`.
- The prediction threshold is 0.5.
- Dataset images and generated outputs are ignored by Git to avoid committing
  large or generated files. Download or prepare data separately and comply
  with the source dataset's licenses and attribution requirements.
- The included model and labels are examples, not evidence of clinical
  performance. Validate any research use with appropriately sourced data and
  qualified domain expertise.
