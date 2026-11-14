"""
======================================================
PROJECT 10: CROP DISEASE DETECTION - TENSORFLOW MODEL
======================================================
Python script to build, train, evaluate, and save the
TensorFlow Keras Convolutional Neural Network (CNN) model:
    `crop_disease_model.keras`

Classes:
    0: Healthy
    1: Leaf Blight
    2: Rust
    3: Early Blight
    4: Septoria
======================================================
"""

import os
import sys

# Configuration
MODEL_PATH = "crop_disease_model.keras"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5

CLASS_NAMES = [
    "Healthy",
    "Leaf Blight",
    "Rust",
    "Early Blight",
    "Septoria"
]


def check_tensorflow():
    """Verifies that TensorFlow and NumPy are installed."""
    try:
        import tensorflow as tf
        import numpy as np
        print(f"[✓] TensorFlow version: {tf.__version__}")
        print(f"[✓] NumPy version: {np.__version__}")
        return tf, np
    except ImportError as e:
        print("\n" + "=" * 60)
        print("❌ TensorFlow is not installed in this Python environment.")
        print("Please install the required packages using:")
        print("    pip install tensorflow numpy pillow")
        print("=" * 60 + "\n")
        sys.exit(1)


def build_crop_disease_cnn(tf, input_shape=(224, 224, 3), num_classes=5):
    """
    Constructs a Deep Convolutional Neural Network (CNN) architecture
    optimized for multi-class plant foliage pathology identification.
    """
    from tensorflow.keras import layers, models

    print("\n[Architecture] Building TensorFlow CNN...")

    model = models.Sequential([
        # Input Layer
        layers.Input(shape=input_shape),

        # Data Augmentation (active during training)
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),

        # Convolutional Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolutional Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolutional Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolutional Block 4
        layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Global Pooling & Fully-Connected Dense Layers
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),

        # Output Softmax Layer (5 classes)
        layers.Dense(num_classes, activation="softmax", name="disease_output")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


def generate_and_save_model(dataset_dir=None):
    """
    Builds and trains the model. If a dataset directory with subfolders
    for each disease is provided, it trains on actual images.
    Otherwise, it synthesizes characteristic leaf training tensors
    to initialize weights and save `crop_disease_model.keras`.
    """
    tf, np = check_tensorflow()

    model = build_crop_disease_cnn(tf, input_shape=(224, 224, 3), num_classes=NUM_CLASSES)

    if dataset_dir and os.path.exists(dataset_dir):
        print(f"\n[Training] Loading dataset from: {dataset_dir}")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical"
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical"
        )

        # Scale pixel inputs
        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

        print("[Training] Fitting model on dataset...")
        model.fit(train_ds, validation_data=val_ds, epochs=15)

    else:
        print("\n[Synthetic Training] Generating representative leaf pathology tensors...")
        # 150 synthetic samples (30 per class) with realistic color/frequency signatures
        samples_per_class = 30
        total_samples = samples_per_class * NUM_CLASSES

        X = np.zeros((total_samples, 224, 224, 3), dtype=np.float32)
        y = np.zeros((total_samples, NUM_CLASSES), dtype=np.float32)

        for c_idx in range(NUM_CLASSES):
            for s in range(samples_per_class):
                idx = c_idx * samples_per_class + s
                # Base leaf green
                tensor = np.random.uniform(0.1, 0.3, (224, 224, 3)).astype(np.float32)
                tensor[:, :, 1] += np.random.uniform(0.4, 0.6)  # High green channel

                if c_idx == 0:  # Healthy
                    tensor[:, :, 0] *= 0.4
                    tensor[:, :, 2] *= 0.3
                elif c_idx == 1:  # Leaf Blight (yellowing/scorched edges)
                    tensor[100:, :, 0] += np.random.uniform(0.3, 0.5)
                    tensor[100:, :, 1] += np.random.uniform(0.2, 0.3)
                elif c_idx == 2:  # Rust (orange/reddish pustules)
                    tensor[50:180, 50:180, 0] += 0.5
                    tensor[50:180, 50:180, 1] += 0.25
                elif c_idx == 3:  # Early Blight (dark brown concentric lesions)
                    tensor[80:150, 80:150, :] = 0.15
                elif c_idx == 4:  # Septoria (scattered small dark spots)
                    for _ in range(25):
                        rx = np.random.randint(10, 210)
                        ry = np.random.randint(10, 210)
                        tensor[rx:rx+8, ry:ry+8, :] = 0.1

                X[idx] = np.clip(tensor, 0.0, 1.0)
                y[idx, c_idx] = 1.0

        print(f"[Training] Training for 5 epochs on initialized foliage tensors...")
        model.fit(X, y, epochs=5, batch_size=16, verbose=1)

    # Save trained model to disk
    print(f"\n[Saving] Writing TensorFlow Keras model to: {MODEL_PATH}")
    model.save(MODEL_PATH)
    print(f"[✓] Model saved successfully as '{MODEL_PATH}'")
    print("\nNext step: Run 'python app.py' to launch the connected web application!")


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_and_save_model(dataset_path)
