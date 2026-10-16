"""ResNet50V2 transfer-learning model with a dropout head (for MC Dropout)."""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2

from backend.config import IMG_SIZE, NUM_CLASSES, DROPOUT_RATE

BACKBONE_NAME = "backbone"


def build_model(num_classes: int = NUM_CLASSES, freeze_base: bool = True) -> tf.keras.Model:
    """Input: raw RGB pixels in 0-255 (scaling to [-1, 1] happens inside the model)."""
    base = ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        name=BACKBONE_NAME,
    )
    base.trainable = not freeze_base

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(inputs)  # ResNetV2 preprocessing
    # training=False keeps BatchNorm in inference mode ALWAYS (also during MC passes
    # and fine-tuning). Only the Dropout layer below is stochastic at MC time.
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc")(x)
    x = layers.Dropout(DROPOUT_RATE, name="mc_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return models.Model(inputs, outputs, name="brain_tumor_resnet50v2")


def unfreeze_top(model: tf.keras.Model, n_layers: int = 30) -> None:
    """Fine-tuning stage: unfreeze the last n layers of the backbone (BatchNorm stays frozen)."""
    base = model.get_layer(BACKBONE_NAME)
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
