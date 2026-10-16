# Architecture

## System

```
React frontend  --HTTP-->  Flask backend
 upload, preview            POST /predict
 result + heatmap             preprocess.py  (resize 224x224, RGB)
 low-confidence warning       mc_dropout.py  (T passes, dropout ON)
                              gradcam.py     (heatmap on last conv map)
                              decision rule  (entropy > threshold -> needs review)
                            GET /health
Model is loaded once at startup from weights/brain_tumor_model.keras
```

## Model

```
Input 224x224x3 (0-255)
 -> Rescaling to [-1, 1]
 -> ResNet50V2 backbone (ImageNet, BatchNorm always in inference mode)
 -> GlobalAveragePooling2D
 -> Dense 256 ReLU
 -> Dropout 0.4   (kept ON for MC passes)
 -> Dense N softmax
```

## Prediction logic

```
for t in 1..T:  p_t = model(x, training=True)
p_mean  = mean(p_t)
class   = argmax(p_mean)
entropy = -sum(p_mean * log p_mean) / log(N)      # normalised to [0, 1]
needs_review = entropy > threshold                # threshold chosen on validation data
```

## Training

1. Stage 1: backbone frozen, train the head (Adam 1e-3)
2. Stage 2: unfreeze the last 30 backbone layers (BatchNorm stays frozen), Adam 1e-5
3. Augmentation on training data only; validation is split from Training; Testing is untouched

## Why BatchNorm stays frozen

If BatchNorm ran in training mode during MC passes, batch statistics would change every pass and
add noise unrelated to dropout. The backbone is therefore called with `training=False`.
