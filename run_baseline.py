import os

import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from data.dataset_loader import load_dataset, CLASS_NAMES
from preprocessing.preprocess import preprocess_dataset
from models.baseline_cnn import build_baseline_cnn


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    confusion = np.array([
        [tn, fp],
        [fn, tp],
    ])

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


# The CIFAKE dataset is inside this repo
DATASET_DIR = "data"
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32
EPOCHS = 3

# 1. Load and label (0 = REAL, 1 = FAKE), with a validation split
train_ds, val_ds, test_ds = load_dataset(DATASET_DIR, batch_size=BATCH_SIZE)
print("Label mapping:", {name: i for i, name in enumerate(CLASS_NAMES)})
print("Train dataset classes:", train_ds.class_names)
print("Validation dataset classes:", val_ds.class_names)
print("Test dataset classes:", test_ds.class_names)

# Quick dataset sanity check: confirm the model is consuming the repo's CIFAKE data
for images, labels in train_ds.take(1):
    flat_labels = labels.numpy().reshape(-1).tolist()
    unique_labels = sorted({int(label) for label in flat_labels})
    print("Sample training labels:", unique_labels)
    assert set(unique_labels).issubset({0, 1}), "Unexpected labels in training batch."

# 2. Preprocess: resize and normalize
train_ds = preprocess_dataset(train_ds, IMAGE_SIZE)
val_ds = preprocess_dataset(val_ds, IMAGE_SIZE)
test_ds = preprocess_dataset(test_ds, IMAGE_SIZE)

# 3. Build the baseline CNN
model = build_baseline_cnn(input_shape=(*IMAGE_SIZE, 3))
model.summary()

# 4. Train and evaluate with real metrics
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)

print("\nFinal validation metrics:")
for key in ["accuracy", "precision", "recall"]:
    if key in history.history:
        print(f"- {key}: {history.history[key][-1]:.4f}")

# 5. Test set evaluation with binary metrics
loss, accuracy, precision, recall = model.evaluate(test_ds, verbose=0)
print("\nTest set evaluation:")
print(f"- loss: {loss:.4f}")
print(f"- accuracy: {accuracy:.4f}")
print(f"- precision: {precision:.4f}")
print(f"- recall: {recall:.4f}")

true_labels = np.concatenate([y for _, y in test_ds], axis=0)
probs = model.predict(test_ds, verbose=0).reshape(-1)
metrics = compute_binary_metrics(true_labels, probs)

print("- f1: {:.4f}".format(metrics["f1"]))
print("- confusion matrix:")
print(metrics["confusion_matrix"])
print("- tn, fp, fn, tp:", metrics["true_negative"], metrics["false_positive"], metrics["false_negative"], metrics["true_positive"])

print("\nPipeline metrics finished.")