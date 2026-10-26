# 🧠 Brain Tumor Detection Using CNN

An AI-powered brain MRI image classification project that uses a Convolutional Neural Network (CNN) to classify brain MRI images into four categories: **Glioma, Meningioma, Pituitary Tumor, and No Tumor**.

The project combines deep learning with a Flask backend and a React frontend to provide an interactive interface for image uploads, model predictions, confidence scores, class probabilities, and uncertainty estimation.

## 📸 Project Preview

### 1. Application Dashboard

![Application Dashboard](screenshots/dashboard.png)

### 2. MRI Image Upload

![MRI Image Upload](screenshots/image_upload.png)

### 3. Prediction Results

![Prediction Results](screenshots/prediction_result.png)

## 🎥 Project Demonstration

Watch the demonstration to see how an MRI image is uploaded, processed by the CNN model, and displayed with its prediction results.

**[▶ Watch the Project Demo](demo/project_demo.mp4)**

> Replace the paths above with your actual screenshots and video. GitHub may show an MP4 as a downloadable file rather than an embedded video player. Alternatively, upload the demo to YouTube and link it here.

## ✨ Features

- **Four-Class Classification:** Classifies MRI images into glioma, meningioma, pituitary tumor, and no-tumor categories.
- **CNN-Based Prediction:** Uses a trained Convolutional Neural Network for image classification.
- **Interactive Web Interface:** Upload and preview MRI images through a React frontend.
- **Prediction Confidence:** Displays the confidence score associated with a model prediction.
- **Class Probabilities:** Shows the model's scores for the supported classes.
- **Uncertainty Estimation:** Uses prediction entropy and a configurable threshold to flag uncertain outputs.
- **Flask REST API:** Connects the frontend to the trained model.
- **Scan History:** Displays recent predictions during the current session.
- **Visual Explanation Support:** Supports heatmap display when the backend provides one.
- **Responsive Design:** Provides a dashboard designed for desktop and mobile screens.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development and inference |
| NumPy | Numerical operations |
| Pillow | Image processing |
| Flask | Backend API |
| React | Frontend interface |
| JavaScript | Frontend functionality |
| CSS | User interface styling |
| Kaggle | Model training and experimentation |
| Git and GitHub | Version control and project hosting |

## 🧠 Model Performance

The model achieved the following results on the test dataset.

| Evaluation Metric | Result |
|---|---:|
| Test Accuracy | 94.3% |
| Accepted Prediction Coverage | 96.9% |
| Accuracy on Accepted Predictions | 96.0% |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Glioma | 99.0% | 80.6% | 88.9% |
| Meningioma | 86.4% | 97.2% | 91.5% |
| No Tumor | 96.2% | 100.0% | 98.0% |
| Pituitary | 97.5% | 99.0% | 98.3% |

**Note:** These are the reported results from the project's test evaluation. Accepted prediction coverage represents the proportion of predictions passing the uncertainty filter. Accuracy on accepted predictions is measured only on that subset. Results may vary on images from other sources.

## 📂 Project Structure

```text
BRAIN_TUMOR_DETECTION_CNN/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── CompareViewer.jsx
│   │   │   ├── ProbabilityBars.jsx
│   │   │   └── UncertaintyGauge.jsx
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   └── package.json
│
├── training/
│
├── weights/
│   ├── brain_tumor_model.keras
│   └── class_names.json
│
├── screenshots/
│   ├── dashboard.png
│   ├── image_upload.png
│   └── prediction_result.png
│
├── demo/
│   └── project_demo.mp4
│
├── app.py
├── requirements.txt
├── start.ps1
└── README.md
```

*This is an example structure. Adjust it to match the files and folders in your actual repository.*

## ⚙️ Installation and Setup

### Prerequisites

- Python installed on your system
- Node.js and npm
- Git
- A compatible trained CNN model

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd BRAIN_TUMOR_DETECTION_CNN
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### Step 2: Create a Python Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Check the Model Files

Ensure that the trained model and class names file are available at the locations expected by your backend:

```text
weights/
├── brain_tumor_model.keras
└── class_names.json
```

If your actual model is stored elsewhere, update the path in the backend configuration accordingly.

### Step 5: Start the Flask Backend

```bash
python app.py
```

Keep the backend terminal running. The frontend is configured to use `http://127.0.0.1:5000` by default unless `VITE_API_URL` is set.

### Step 6: Start the React Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local development URL displayed in the terminal.

## 🔍 How It Works

1. **Image Upload:** The user selects a brain MRI image through the frontend.
2. **Image Preprocessing:** The backend prepares the image in the format expected by the trained model.
3. **CNN Inference:** The model processes the image and generates classification scores.
4. **Prediction:** The application identifies the class with the highest model score.
5. **Confidence Analysis:** The interface displays the prediction confidence and available class probabilities.
6. **Uncertainty Estimation:** An entropy-based threshold can flag predictions that require additional caution.
7. **Result Display:** The frontend displays the prediction and any additional information returned by the backend.

## 📊 Uncertainty Estimation

The project uses prediction entropy to estimate uncertainty in the model's probability distribution.

An uncertainty threshold is used to flag certain predictions for cautious interpretation. This provides an additional signal beyond the predicted class alone.

The uncertainty score is a model-based indicator and does not establish whether a prediction is medically correct.

## 🚀 Future Improvements

- Integrate Grad-CAM to highlight image regions that influence predictions.
- Improve recall for classes with comparatively weaker results.
- Evaluate the model on independent MRI datasets.
- Add confusion matrix and per-class performance visualizations.
- Generate downloadable prediction reports.
- Improve API validation and error handling.
- Deploy the application using an appropriate hosting platform.
- Add automated testing for preprocessing and inference.

## 🔮 Applications

- Educational demonstrations of CNN-based image classification.
- Experimentation with medical image preprocessing and deep learning.
- Exploring uncertainty estimation in machine learning.
- Demonstrating the integration of an ML model with a web application.

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. It is **not a medical diagnostic tool** and must not be used to diagnose or rule out a brain tumor. Model predictions can be incorrect and must not replace evaluation by a qualified medical professional.

## 👨‍💻 Author

Developed as an Artificial Intelligence and Machine Learning project to explore deep learning, brain MRI image classification, model evaluation, uncertainty estimation, and full-stack AI application development.
