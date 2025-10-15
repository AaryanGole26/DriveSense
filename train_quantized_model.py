# ============================================
# train_quantized_model.py
# ============================================
# Author: Aaryan Gole
# Description: CNN + Quantization-Aware Training + INT8 TFLite export
# ============================================

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import cv2
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
data = pd.read_csv('dataset/labels.csv')

images, labels = [], []

for _, row in data.iterrows():
    img_path = os.path.join('dataset', row['image_path'])
    print(f"Trying to load: {img_path}")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Failed to load: {img_path}")
        continue
    img = cv2.resize(img, (48, 48)) / 255.0
    images.append(img)
    labels.append(row['label'])

X = np.array(images).reshape(-1, 48, 48, 1)

# Encode labels if not numeric
if not np.issubdtype(type(labels[0]), np.number):
    le = LabelEncoder()
    y = le.fit_transform(labels)
else:
    y = np.array(labels)

# -----------------------------
# 2️⃣ Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3️⃣ Define CNN Model
# -----------------------------
def build_model():
    inputs = tf.keras.Input(shape=(48, 48, 1))
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(
        64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

model = build_model()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n🧠 Training Base Model...")
model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# -----------------------------
# 4️⃣ Apply Quantization-Aware Training
# -----------------------------
print("\n⚙️  Applying Quantization-Aware Training...")

quantize_model = tfmot.quantization.keras.quantize_model
q_aware_model = quantize_model(model)

q_aware_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n🔁 Fine-tuning Quantized Model...")
q_aware_model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# -----------------------------
# 5️⃣ Convert to TFLite (Full INT8)
# -----------------------------
print("\n🧩 Converting to TFLite INT8 Model...")

def representative_data_gen():
    for i in range(min(100, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

# -----------------------------
# 6️⃣ Save Model
# -----------------------------
output_path = 'distress_model.tflite'
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f"\n✅ Quantized model successfully saved as: {output_path}")

# -----------------------------
# 7️⃣ Evaluate Model
# -----------------------------
loss, acc = q_aware_model.evaluate(X_test, y_test, verbose=0)
print(f"\n📊 Final Model Accuracy: {acc:.4f}, Loss: {loss:.4f}")
