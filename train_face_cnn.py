"""
train_face_cnn.py - train our own facial-emotion CNN on FER-2013 (folder format).

Expected layout (Kaggle "msambare/fer2013"):
    data/fer2013/train/<emotion>/*.png
    data/fer2013/test/<emotion>/*.png       (angry, disgust, fear, happy, neutral, sad, surprise)

    python train_face_cnn.py --epochs 60

Saves saved_models/face_cnn/model.keras and class_names.json (class order = sorted
folder names, exactly as Keras assigned the integer labels). 10% of train is used as
validation for early stopping; the official test folder is only used for the final report.
"""
import os
import json
import argparse

import numpy as np
import tensorflow as tf
import keras                      # Keras 3 (installed with TensorFlow >= 2.16)
from keras import layers
from sklearn.utils.class_weight import compute_class_weight

import config
from metrics_utils import save_results

OUT_DIR = os.path.join(config.MODELS_DIR, "face_cnn")


def build_face_cnn(n_classes, size=config.FACE_IMG_SIZE):
    inp = keras.Input(shape=(size, size, 1))
    # Augmentation layers are only active during training (Keras disables them at inference).
    x = layers.RandomFlip("horizontal")(inp)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)
    x = layers.Rescaling(1.0 / 255)(x)                 # model accepts raw 0..255 pixels
    for filters, drop in [(64, 0.25), (128, 0.25), (256, 0.3)]:
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(drop)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out, name="face_cnn")
    model.compile(optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train a face-emotion CNN on FER-2013.")
    parser.add_argument("--data-dir", default=config.FER_DIR)
    parser.add_argument("--epochs", type=int, default=config.FACE_EPOCHS)
    parser.add_argument("--batch", type=int, default=config.FACE_BATCH)
    args = parser.parse_args()
    keras.utils.set_random_seed(config.SEED)

    common = dict(color_mode="grayscale", image_size=(config.FACE_IMG_SIZE,) * 2,
                  batch_size=args.batch, label_mode="int", seed=config.SEED)
    train_ds = keras.utils.image_dataset_from_directory(os.path.join(args.data_dir, "train"),
                                                        validation_split=0.1, subset="training", **common)
    val_ds = keras.utils.image_dataset_from_directory(os.path.join(args.data_dir, "train"),
                                                      validation_split=0.1, subset="validation", **common)
    test_ds = keras.utils.image_dataset_from_directory(os.path.join(args.data_dir, "test"),
                                                       shuffle=False, **common)
    class_names = train_ds.class_names                 # alphabetical folder names
    print("classes:", class_names)

    y_train = np.concatenate([y.numpy() for _, y in train_ds])
    weights = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=y_train)
    class_weight = dict(enumerate(weights.astype(float)))

    autotune = tf.data.AUTOTUNE
    train_ds, val_ds = train_ds.prefetch(autotune), val_ds.prefetch(autotune)

    model = build_face_cnn(len(class_names))
    model.summary()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=class_weight,
                     callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, mode="max",
                                                              restore_best_weights=True),
                                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)])

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save(os.path.join(OUT_DIR, "model.keras"))
    with open(os.path.join(OUT_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f)
    print(f"saved model + class_names to {OUT_DIR}")

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = model.predict(test_ds, verbose=0).argmax(axis=1)
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    save_results("face_cnn", y_true, y_pred, history=history, class_names=class_names,
                 extra={"task": "face"})


if __name__ == "__main__":
    main()
