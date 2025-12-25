import os
import json
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split



class DatasetLoader:
    def __init__(self, csv_path, images_dirs, img_size=(128, 128)):
        self.csv_path = csv_path
        self.images_dirs = images_dirs
        self.img_size = img_size
        self.extensions = [".jpg", ".jpeg", ".png"]

    def _resolve_image_path(self, image_id):
        for d in self.images_dirs:
            for ext in self.extensions:
                p = os.path.join(d, image_id + ext)
                if os.path.exists(p):
                    return p
        return None

    def load(self):
        df = pd.read_csv(self.csv_path)
        df = df[df["label"] != "Not sure"].reset_index(drop=True)

        images, labels = [], []

        for _, row in df.iterrows():
            img_path = self._resolve_image_path(str(row["image"]))
            if img_path is None:
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            img = img.resize(self.img_size)
            arr = np.asarray(img, dtype=np.float32) / 255.0 

            images.append(arr)
            labels.append(row["label"])

        return np.array(images), np.array(labels)



class LabelEncoder:
    def __init__(self):
        self.class_names = []

    def fit_transform(self, labels):
        self.class_names = sorted(set(labels))
        label_to_id = {l: i for i, l in enumerate(self.class_names)}
        return np.array([label_to_id[l] for l in labels])

    def save(self, path="labels.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.class_names, f, ensure_ascii=False, indent=2)



class DataAugmentation:
    def __init__(self):
        self.augmenter = tf.keras.Sequential([
            layers.RandomZoom(0.05),
            layers.RandomContrast(0.10),
        ])



class ClothingCNNModel:
    def __init__(self, input_shape, num_classes, augmenter=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.augmenter = augmenter

    def build(self):
        layers_list = [layers.Input(shape=self.input_shape)]

        if self.augmenter is not None:
            layers_list.append(self.augmenter.augmenter)

        layers_list += [
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),

            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(self.num_classes, activation="softmax")
        ]

        model = models.Sequential(layers_list)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(3e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model



class Trainer:
    def __init__(self, model):
        self.model = model

    def train(self, x_train, y_train, x_val, y_val, epochs=40):
        callbacks = [
            EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1)
        ]

        return self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            callbacks=callbacks
        )

    def save(self, path="model.keras"):
        self.model.save(path)



if __name__ == "__main__":

    CSV_PATH = r"clothing_dataset_full\images.csv"
    IMAGE_DIRS = [
        r"D:\KP_Prog1\clothing_dataset_full\images_compressed"
    ]
    IMG_SIZE = (128, 128)

    loader = DatasetLoader(CSV_PATH, IMAGE_DIRS, IMG_SIZE)
    X, y_text = loader.load()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    encoder.save("labels.json")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.15,
        random_state=42
    )

    augmenter = DataAugmentation()

    cnn = ClothingCNNModel(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        num_classes=len(encoder.class_names),
        augmenter=augmenter
    )
    model = cnn.build()
    model.summary()

    trainer = Trainer(model)
    trainer.train(X_train, y_train, X_val, y_val)

    trainer.save("model.keras")
