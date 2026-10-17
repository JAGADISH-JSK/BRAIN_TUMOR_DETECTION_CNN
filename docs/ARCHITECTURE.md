# Architecture

This page explains how the project works, from training to the web app.
(GitHub draws the diagrams below automatically.)

## 1. System overview

```mermaid
flowchart LR
    U[User] --> F[React frontend]
    F -->|"POST /predict (image)"| B[Flask API]
    B --> P[Preprocess to 224x224 RGB]
    P --> M[ResNet50V2 with dropout head]
    M -->|"T = 20 passes, dropout ON"| E[Mean probabilities and entropy]
    M --> G[Grad-CAM heatmap]
    E --> D{Entropy above threshold?}
    D -->|Yes| R[Needs expert review]
    D -->|No| C[Class and confidence]
    R --> J[JSON response]
    C --> J
    G --> J
    J --> F
```

The model is loaded once when the Flask server starts, not on every request.

## 2. Training and evaluation flow

```mermaid
flowchart TD
    A[MRI dataset: Training and Testing folders] --> B[Hold out 15 percent of Training for validation]
    B --> C[Augment training images only]
    C --> D[Stage 1: train the new head, backbone frozen]
    D --> E[Stage 2: fine-tune the top 30 backbone layers]
    E --> F[Save the best model by validation accuracy]
    F --> G[Choose the entropy threshold on validation data]
    G --> H[Report final results on the Testing folder]
```

## 3. Model

```
Input 224 x 224 x 3 (pixel values 0-255)
  -> Rescaling to [-1, 1]
  -> ResNet50V2 backbone (ImageNet weights, BatchNorm always in inference mode)
  -> GlobalAveragePooling2D
  -> Dense 256, ReLU
  -> Dropout 0.4        (stays ON during the MC passes)
  -> Dense N, softmax   (one output per class)
```

## 4. Prediction logic (Monte Carlo Dropout)

```
for t in 1..T:  p_t = model(image, training=True)      # dropout active
p_mean       = mean(p_t)
predicted    = argmax(p_mean)
uncertainty  = entropy(p_mean) / log(N)                # scaled to 0..1
needs_review = uncertainty > threshold                 # threshold chosen on validation data
```

## 5. Why BatchNorm stays frozen

If BatchNorm ran in training mode during the MC passes, its statistics would change on every pass
and add noise that has nothing to do with dropout. The backbone is therefore always called with
`training=False`.

## 6. Code map

| Path | Role |
|---|---|
| `backend/model/build.py` | Builds the ResNet50V2 model, fine-tuning helper |
| `backend/model/mc_dropout.py` | T stochastic passes and the entropy score |
| `backend/model/gradcam.py` | Heatmap on the last convolutional map |
| `backend/app.py` | Flask API: `/predict`, `/health` |
| `training/train.py` | Two-stage training |
| `training/evaluate.py` | Metrics, accuracy vs. coverage, uncertainty plots |
| `frontend/` | React upload page and result view |
