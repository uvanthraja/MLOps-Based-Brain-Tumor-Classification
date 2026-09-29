import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\ASUS\My_Learning\medical mlops project"

TRAIN_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "split_dataset",
    "train"
)

VALIDATION_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "split_dataset",
    "validation"
)

TEST_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "Testing"
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

NUM_CLASSES = 4

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

INITIAL_EPOCHS = 15

FINE_TUNE_EPOCHS = 15

INITIAL_LR = 1e-4

FINE_TUNE_LR = 1e-5


# ============================================================
# 3. PRINT CONFIGURATION
# ============================================================

print("=" * 70)
print("MedOpsAI - Brain Tumor MRI Classification")
print("DenseNet121 Training Pipeline")
print("=" * 70)

print("\nProject Directory:")
print(PROJECT_DIR)

print("\nTraining Directory:")
print(TRAIN_DIR)

print("\nValidation Directory:")
print(VALIDATION_DIR)

print("\nTesting Directory:")
print(TEST_DIR)


# ============================================================
# 4. CHECK DIRECTORIES
# ============================================================

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training directory not found:\n{TRAIN_DIR}"
    )

if not os.path.exists(VALIDATION_DIR):
    raise FileNotFoundError(
        f"Validation directory not found:\n{VALIDATION_DIR}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Testing directory not found:\n{TEST_DIR}"
    )


# ============================================================
# 5. DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,

    rotation_range=15,

    width_shift_range=0.10,

    height_shift_range=0.10,

    zoom_range=0.10,

    shear_range=0.05,

    horizontal_flip=True,

    brightness_range=[0.90, 1.10],

    fill_mode="nearest"
)


# ============================================================
# 6. VALIDATION AND TEST PREPROCESSING
# ============================================================

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)


# ============================================================
# 7. CREATE DATA GENERATORS
# ============================================================

print("\nLoading training data...")

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    classes=CLASS_NAMES,

    shuffle=True
)


print("\nLoading validation data...")

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    classes=CLASS_NAMES,

    shuffle=False
)


print("\nLoading testing data...")

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    classes=CLASS_NAMES,

    shuffle=False
)


print("\nDataset Summary")
print("-" * 50)

print(
    "Training images   :",
    train_generator.samples
)

print(
    "Validation images :",
    validation_generator.samples
)

print(
    "Testing images    :",
    test_generator.samples
)

print(
    "Classes           :",
    CLASS_NAMES
)

print(
    "Class mapping     :",
    train_generator.class_indices
)


# ============================================================
# 8. LOAD DENSENET121
# ============================================================

print("\nLoading DenseNet121 pretrained on ImageNet...")

base_model = DenseNet121(
    weights="imagenet",

    include_top=False,

    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)


# ============================================================
# 9. FREEZE BASE MODEL
# ============================================================

base_model.trainable = False


# ============================================================
# 10. BUILD CLASSIFICATION HEAD
# ============================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.30)(x)

x = Dense(
    256,
    activation="relu"
)(x)

x = Dropout(0.30)(x)

output = Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


# ============================================================
# 11. CREATE MODEL
# ============================================================

model = Model(
    inputs=base_model.input,
    outputs=output
)


# ============================================================
# 12. COMPILE MODEL - STAGE 1
# ============================================================

model.compile(
    optimizer=Adam(
        learning_rate=INITIAL_LR
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# 13. CALLBACKS
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "densenet121_brain_tumor_best.keras"
)

final_model_path = os.path.join(
    MODEL_DIR,
    "densenet121_brain_tumor_final.keras"
)

checkpoint = ModelCheckpoint(
    best_model_path,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",

    factor=0.2,

    patience=2,

    min_lr=1e-7,

    verbose=1
)


# ============================================================
# 14. STAGE 1 - TRANSFER LEARNING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1 - TRANSFER LEARNING")
print("=" * 70)

history_stage1 = model.fit(
    train_generator,

    validation_data=validation_generator,

    epochs=INITIAL_EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# 15. STAGE 2 - FINE TUNING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 - FINE TUNING")
print("=" * 70)


base_model.trainable = True


# Freeze first 70% of DenseNet layers

freeze_until = int(
    len(base_model.layers) * 0.70
)

for layer in base_model.layers[:freeze_until]:
    layer.trainable = False


print(
    "\nTotal DenseNet layers:",
    len(base_model.layers)
)

print(
    "Frozen layers:",
    freeze_until
)

print(
    "Trainable layers:",
    len(base_model.layers) - freeze_until
)


# ============================================================
# 16. COMPILE FOR FINE TUNING
# ============================================================

model.compile(
    optimizer=Adam(
        learning_rate=FINE_TUNE_LR
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# 17. FINE-TUNE MODEL
# ============================================================

history_stage2 = model.fit(
    train_generator,

    validation_data=validation_generator,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# 18. SAVE FINAL MODEL
# ============================================================

model.save(
    final_model_path
)

print("\nFinal model saved:")
print(final_model_path)


# ============================================================
# 19. COMBINE TRAINING HISTORY
# ============================================================

history = {}

for key in history_stage1.history:

    history[key] = (
        history_stage1.history[key]
        +
        history_stage2.history.get(
            key,
            []
        )
    )


# ============================================================
# 20. SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    MODEL_DIR,
    "densenet121_training_history.json"
)

with open(
    history_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        history,
        file,
        indent=4
    )

print(
    "\nTraining history saved:"
)

print(history_path)


# ============================================================
# 21. PLOT ACCURACY
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "DenseNet121 Training and Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.tight_layout()

accuracy_plot_path = os.path.join(
    MODEL_DIR,
    "training_accuracy.png"
)

plt.savefig(
    accuracy_plot_path,
    dpi=200
)

plt.close()


# ============================================================
# 22. PLOT LOSS
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    history["loss"],
    label="Training Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "DenseNet121 Training and Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

loss_plot_path = os.path.join(
    MODEL_DIR,
    "training_loss.png"
)

plt.savefig(
    loss_plot_path,
    dpi=200
)

plt.close()


# ============================================================
# 23. LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

best_model = tf.keras.models.load_model(
    best_model_path
)


# ============================================================
# 24. TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy = best_model.evaluate(
    test_generator,

    verbose=1
)

print(
    f"\nTest Accuracy: {test_accuracy * 100:.2f}%"
)

print(
    f"Test Loss    : {test_loss:.4f}"
)


# ============================================================
# 25. TEST PREDICTIONS
# ============================================================

test_generator.reset()

predictions = best_model.predict(
    test_generator,

    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = test_generator.classes


# ============================================================
# 26. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print(
    f"\nCalculated Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# 27. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,

    y_pred,

    target_names=CLASS_NAMES,

    digits=4,

    zero_division=0
)

print("\nClassification Report")
print("-" * 70)

print(report)


# ============================================================
# 28. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix")
print("-" * 70)

print(cm)


# ============================================================
# 29. SAVE FINAL RESULTS
# ============================================================

results = {
    "model": "DenseNet121",

    "dataset": "Brain Tumor MRI",

    "classes": CLASS_NAMES,

    "test_images": int(
        test_generator.samples
    ),

    "test_accuracy": float(
        accuracy
    ),

    "test_loss": float(
        test_loss
    )
}

results_path = os.path.join(
    MODEL_DIR,
    "final_results.json"
)

with open(
    results_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ============================================================
# 30. COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING AND EVALUATION COMPLETED")
print("=" * 70)

print(
    f"\nFinal Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print("\nSaved files:")

print(
    "- densenet121_brain_tumor_best.keras"
)

print(
    "- densenet121_brain_tumor_final.keras"
)

print(
    "- densenet121_training_history.json"
)

print(
    "- training_accuracy.png"
)

print(
    "- training_loss.png"
)

print(
    "- final_results.json"
)

print("\nMedOpsAI training pipeline completed.")