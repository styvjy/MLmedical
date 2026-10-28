# Chest X-Ray Pneumonia Image Classifier

An end-to-end Computer Vision and Deep Learning system for classifying pediatric and adult chest radiographs into **NORMAL** and **PNEUMONIA** using **MobileNetV2 Transfer Learning**.

Includes an automated dataset verification engine, early stopping checkpoint pipeline, independent test set evaluation, and a modern dark glassmorphic web diagnostic portal.

---

## 1. Project Architecture & Directory Layout

```
c:\satya project AI\
├── dataset/                          # Dataset root directory
│   ├── train/                        # Training split
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/                          # Validation split (for early stopping)
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/                         # Independent test split (for final metrics)
│       ├── NORMAL/
│       └── PNEUMONIA/
├── models/                           # Trained weights and evaluation metadata
│   ├── mobilenetv2_pneumonia_best.pth  # Best checkpoint saved during training
│   ├── class_indices.json            # Exact mapping {"NORMAL": 0, "PNEUMONIA": 1}
│   ├── training_history.json         # Loss and accuracy progression
│   ├── evaluation_metrics.json       # Validated test metrics & confusion matrix
│   └── confusion_matrix.png          # Visual confusion matrix heatmap
├── src/
│   ├── __init__.py
│   ├── config.py                     # Paths, device selector, hyperparameters
│   ├── data_loader.py                # Preprocessing, data augmentation, class weighting
│   ├── model.py                      # MobileNetV2 architecture with custom classification head
│   ├── train.py                      # Real ML training with early stopping & checkpointing
│   ├── evaluate.py                   # Test set evaluation (Accuracy, Recall, Precision, F1, ROC-AUC)
│   └── predict.py                    # Inference engine with exact transform matching & validation
├── web/
│   ├── app.py                        # Flask backend with REST API and sample endpoints
│   ├── templates/
│   │   └── index.html                # Semantic HTML5 diagnostic workbench
│   └── static/
│       ├── css/style.css             # Glassmorphic dark theme stylesheet
│       ├── js/main.js                # Drag & drop, real-time prediction, confidence gauges
│       └── sample_images/            # Pre-generated sample radiographs for instant testing
├── tests/
│   ├── test_environment.py           # Dependency and hardware verification
│   ├── test_data_loader.py           # Transforms, RGB conversions, dataset verification
│   ├── test_model.py                 # Forward pass shapes and checkpoint roundtrip
│   ├── test_inference.py             # Image validation, corrupt file rejection, prediction format
│   └── test_web_api.py               # REST API endpoints and error handling
├── download_dataset.py               # Dataset download & restructuring helper
└── requirements.txt                  # Python dependencies
```

---

## 2. Requirements & Setup on Windows in VS Code

### Prerequisites
- Python 3.10+ (Tested on Python 3.14 64-bit on Windows)
- Windows PowerShell or Command Prompt

### Step 1: Open Project in VS Code
Open VS Code, press `Ctrl+O` (or `File > Open Folder...`), and select:
```text
C:\satya project AI
```

### Step 2: Open Terminal & Create Virtual Environment (Optional but recommended)
Open the integrated terminal in VS Code (`Ctrl + ~`) and run:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 3. Dataset Download and Preparation

The model is designed for the standard **Kaggle Chest X-Ray Images (Pneumonia)** dataset by Paul Mooney.

### Option A: Automatic Download via KaggleHub
If you have `kagglehub` installed:
```powershell
python download_dataset.py --auto
```

### Option B: Manual Download
1. Download the dataset archive (~1.15 GB) from Kaggle:
   [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Extract the archive into the `dataset/` directory so the folder structure is:
   ```
   dataset/
     train/
       NORMAL/
       PNEUMONIA/
     val/
       NORMAL/
       PNEUMONIA/
     test/
       NORMAL/
       PNEUMONIA/
   ```
3. If extracted inside a nested folder (e.g. `dataset/chest_xray/chest_xray/train`), run:
   ```powershell
   python download_dataset.py --reorganize
   ```
4. Verify dataset readiness at any time:
   ```powershell
   python download_dataset.py
   ```

---

## 4. Model Training

The training script verifies dataset existence before execution. If the dataset is missing, it displays step-by-step guidance rather than crashing.

### Run Training:
```powershell
python -m src.train
```

### Training Features:
- **Hardware Configurable:** Automatically uses CUDA GPU if available; otherwise utilizes optimized CPU threads. You can also specify `--device cpu` or `--device cuda`.
- **Class Balancing:** Automatically computes inverse-frequency class weights for `CrossEntropyLoss` to counteract class imbalance.
- **Data Augmentation:** Applies random horizontal flips, random rotations (10°), and subtle color jitter on training images.
- **Early Stopping:** Tracks validation loss with a patience of 4 epochs.
- **Model Checkpointing:** Saves the best performing model weights to `models/mobilenetv2_pneumonia_best.pth`.
- **Training History:** Records training loss, training accuracy, validation loss, and validation accuracy to `models/training_history.json`.

---

## 5. Independent Test Set Evaluation

Run evaluation on the unseen test set to produce verified metrics:
```powershell
python -m src.evaluate
```

This generates and displays:
- Overall Accuracy
- Precision (Positive Predictive Value)
- Recall / Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)
- F1-Score
- ROC-AUC Score
- Complete Confusion Matrix (`[[TN, FP], [FN, TP]]`)
- Saved metrics file: `models/evaluation_metrics.json`
- Saved confusion matrix plot: `models/confusion_matrix.png`

---

## 6. Launching the Web Application

The interactive web portal allows drag-and-drop radiograph analysis, immediate testing with pre-loaded sample images, and live visualization of prediction probabilities.

### Start the Server:
```powershell
python web/app.py
```

### Access in Browser:
Open your web browser (Chrome, Edge, Firefox) and navigate to:
```text
http://127.0.0.1:5000
```

### Key UI Features:
1. **Dynamic Hardware & Model Indicators:** Shows live active device (`CPU` or `CUDA`) and model status.
2. **Drag & Drop Workbench:** Upload any chest radiograph (`.jpeg`, `.png`, `.bmp`, `.webp`) with instant image preview.
3. **One-Click Quick Test:** Click on pre-loaded sample images (`sample_normal.jpeg` or `sample_pneumonia.jpeg`) for instant evaluation.
4. **Interactive Prediction Meter:** Color-coded diagnostic badge (`NORMAL` in Emerald Green, `PNEUMONIA` in Rose Red) and calibrated confidence percentages.
5. **Real-time Probability Distribution:** Live progress bars displaying individual class probabilities.
6. **Graceful Status Handling:** If the model has not been trained yet, the portal clearly notifies the user with instructions to run `python -m src.train`.

---

## 7. Running the Automated Test Suite

Run the complete test suite (16 unit and integration tests):
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Tested modules:
- `test_environment.py`: Verifies dependencies (PyTorch, TorchVision, Scikit-learn, Pillow, Flask) and hardware detection.
- `test_data_loader.py`: Verifies RGB grayscale conversions, image tensor transforms, and dataset verification.
- `test_model.py`: Verifies MobileNetV2 architecture, tensor shapes, and checkpoint saving/loading.
- `test_inference.py`: Verifies image validation, corrupt file rejection, empty file rejection, and output formatting.
- `test_web_api.py`: Verifies HTTP endpoints (`/`, `/api/status`, `/api/predict`, `/api/samples`), invalid file handling, and response schemas.

---

## 8. Machine Learning Pipeline Mechanics

1. **Preprocessing Alignment:**
   Both training and inference use identical MobileNetV2 normalization:
   - Resized to `(224, 224)`
   - Scaled using standard ImageNet distribution: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
   - Converted single-channel grayscale or RGBA X-rays to 3-channel RGB.

2. **Class Indexing Safeguard:**
   Class index mapping is explicitly serialized to `models/class_indices.json` during data loading (`NORMAL: 0`, `PNEUMONIA: 1`) to eliminate label reversal bugs during inference.

3. **Data Leakage Prevention:**
   The training pipeline strictly isolates the `train/`, `val/`, and `test/` splits. Data augmentation is applied only to the training set; validation and test sets receive deterministic evaluation normalization.

4. **Inference Caching:**
   The `PneumoniaPredictor` loads model weights once into memory on startup and caches them, ensuring fast per-image inference (~15–40 ms on CPU) without retraining or reloading weights on every HTTP request.

---

## 9. Disclaimer

> **Research & Educational Prototype Only:** This system is developed for educational and experimental machine learning purposes. It is not an FDA-approved medical device and must never be used as a primary diagnostic tool in clinical or hospital settings.
