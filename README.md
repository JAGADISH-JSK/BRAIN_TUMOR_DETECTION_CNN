# 🧠 Brain Tumor Detection Using CNN

An AI-powered web application for detecting brain tumors from MRI scans using a Convolutional Neural Network (CNN). The project combines deep learning inference with a Flask backend and a React frontend to provide an interactive experience for image upload, prediction, confidence scoring, and uncertainty estimation.

<p align="center">
  <img src="screenshots/glioma_2x.png" alt="Brain tumor detection dashboard" width="900" />
</p>

## Overview

This project is designed to classify brain MRI images into four categories:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

It aims to demonstrate how a trained CNN can be integrated into a real-world web application for medical image classification, while also exposing model uncertainty to help users interpret results more carefully.

## Features

- Four-class brain tumor classification
- CNN-based prediction pipeline using TensorFlow/Keras
- Easy MRI image upload from the frontend
- Prediction confidence display
- Per-class probability visualization
- Uncertainty estimation using entropy-based filtering
- Flask REST API for model inference
- React dashboard for prediction results
- Support for heatmap visualization and result comparison tools
- Responsive UI for desktop and mobile devices

## Demo

Watch the application in action:

- [Project Demo](demo/project_demo.mp4)

## Tech Stack

| Layer | Technologies |
| --- | --- |
| Deep learning | TensorFlow, Keras |
| Data handling | NumPy, Pandas, Pillow |
| Backend | Flask, Flask-CORS |
| Frontend | React, JavaScript, CSS |
| Model evaluation | scikit-learn, Matplotlib |
| Development | Python, Git, GitHub |

## Model Performance

The trained model achieved the following evaluation results on the test dataset:

| Metric | Value |
| --- | ---: |
| Test Accuracy | 94.3% |
| Accepted Prediction Coverage | 96.9% |
| Accuracy on Accepted Predictions | 96.0% |

### Classification Report

| Class | Precision | Recall | F1-Score |
| --- | ---: | ---: | ---: |
| Glioma | 99.0% | 80.6% | 88.9% |
| Meningioma | 86.4% | 97.2% | 91.5% |
| No Tumor | 96.2% | 100.0% | 98.0% |
| Pituitary | 97.5% | 99.0% | 98.3% |

> These metrics are reported from the model evaluation and should be interpreted as research-oriented results, not clinical diagnosis metrics.

## Project Structure

```text
BRAIN_TUMOR_DETECTION_CNN/
├── backend/
│   ├── model/
│   ├── utils/
│   ├── __init__.py
│   ├── app.py
│   └── config.py
├── data/
├── demo/
│   └── project_demo.mp4
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── screenshots/
│   ├── glioma_2x.png
│   ├── meningioma_2x.png
│   └── pituitary_2x.png
├── training/
├── weights/
│   ├── brain_tumor_model.keras
│   └── class_names.json
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── start.ps1
├── app.py
└── make_small.py
```

## Installation

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm
- Git
- A trained model file in the `weights/` directory

### 1) Clone the repository

```bash
git clone https://github.com/JAGADISH-JSK/BRAIN_TUMOR_DETECTION_CNN.git
cd BRAIN_TUMOR_DETECTION_CNN
```

### 2) Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

- Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

- macOS/Linux:

```bash
source venv/bin/activate
```

### 3) Install backend dependencies

```bash
pip install -r requirements.txt
```

### 4) Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 5) Verify model files

Ensure that the trained CNN weights and class labels are available:

```text
weights/
├── brain_tumor_model.keras
└── class_names.json
```

If your model is stored in a different path, update the backend configuration accordingly.

## Running the Application

### Option 1: Use the project launcher

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

This script helps launch the backend and frontend together.

### Option 2: Run manually

Start the Flask backend:

```bash
python app.py
```

In a new terminal, start the frontend:

```bash
cd frontend
npm run dev
```

Then open the local URL shown in the terminal, typically:

```text
http://localhost:5173
```

## How It Works

1. User uploads a brain MRI image from the frontend.
2. The backend preprocesses the image to match the model input format.
3. The CNN predicts class probabilities for all supported categories.
4. The highest probability is selected as the main prediction.
5. Confidence and uncertainty metrics are computed.
6. The frontend displays the class label, confidence, probability distribution, and uncertainty information.

## API Usage

The backend exposes an inference endpoint for image-based prediction.

### Endpoint

```http
POST /predict
```

### Example request using Python

```python
import requests

url = "http://127.0.0.1:5000/predict"
files = {"image": open("sample.jpg", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

### Example response

```json
{
  "label": "glioma",
  "confidence": 0.91,
  "uncertainty": 0.12,
  "threshold": 0.35,
  "needs_review": false,
  "probabilities": {
    "glioma": 0.91,
    "meningioma": 0.03,
    "notumor": 0.02,
    "pituitary": 0.04
  }
}
```

## Uncertainty Estimation

This project includes entropy-based uncertainty estimation to help flag low-confidence or ambiguous predictions. The threshold can be configured to control when a prediction should be marked for review.

```python
ENTROPY_THRESHOLD = float(os.environ.get("ENTROPY_THRESHOLD", 0.35))
```

This is an auxiliary signal for interpretation and not a clinical measure of certainty.

## Future Improvements

- Add Grad-CAM or saliency map visualizations
- Improve recall for difficult classes
- Evaluate on more diverse MRI datasets
- Add confusion matrix and analytics dashboards
- Improve API validation and logging
- Deploy the app to a cloud hosting platform
- Add automated tests for preprocessing and inference

## Use Cases

- Educational deep learning projects
- MRI image classification demos
- AI + healthcare experimentation
- Full-stack ML application prototyping

## Disclaimer

This project is intended for educational and research purposes only. It is not a medical diagnostic tool and should not be used to diagnose or rule out brain tumors. Any prediction should be reviewed by a qualified medical professional.

## Author

This project was developed to explore CNN-based medical image classification, uncertainty estimation, and full-stack deployment of an AI model.

## License

This project is available under the [MIT License](LICENSE).

## Project Links

- GitHub Repository: https://github.com/JAGADISH-JSK/BRAIN_TUMOR_DETECTION_CNN
- Demo Folder: `demo/project_demo.mp4`
- Model Weights: `weights/brain_tumor_model.keras`

If you want, I can also make this README more portfolio-style, more academic, or more concise for GitHub.
