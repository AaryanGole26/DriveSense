import cv2
import numpy as np
import tensorflow as tf
from pygame import mixer
import time
import os
import random

# ============================================
# WELLNESS MONITOR – Emotion & Distress Detection (OpenCV Version)
# ============================================

# 🎵 Initialize Pygame & Songs
try:
    mixer.init()
    songs_dir = 'songs'
    song_files = [os.path.join(songs_dir, f) for f in os.listdir(songs_dir) if f.endswith('.mp3')]
    if len(song_files) < 5:
        raise FileNotFoundError(f"Found only {len(song_files)} MP3 files in {songs_dir}. Expected at least 5.")
    for song in song_files:
        if not os.path.exists(song):
            raise FileNotFoundError(f"{song} not found.")
    print("✓ Pygame and songs initialized successfully.")
except Exception as e:
    print(f"✗ Error initializing Pygame or loading music: {e}")
    exit()

# 🧠 Load Quantized or Float Model
try:
    interpreter = tf.lite.Interpreter(model_path='distress_model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]['dtype']
    is_quantized = (input_dtype == np.uint8)
    if is_quantized:
        input_scale, input_zero_point = input_details[0]['quantization']
        output_scale, output_zero_point = output_details[0]['quantization']
        print(f"✓ Quantized INT8 model loaded (scale: {input_scale:.6f}, zero_point: {input_zero_point})")
    else:
        print("✓ Float32 model loaded")
    expected_shape = input_details[0]['shape']
    print(f"✓ Model input shape: {expected_shape}")
except Exception as e:
    print(f"✗ Error loading TFLite model: {e}")
    exit()

# 🧍 Initialize OpenCV Face Detector
try:
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_detector.empty():
        raise Exception("Failed to load Haar Cascade")
    print("✓ OpenCV face detector initialized.")
except Exception as e:
    print(f"✗ Error initializing face detector: {e}")
    exit()

# =====================
# Configuration
# =====================
THRESHOLD = 0.5
COOLDOWN = 10
PLAY_DURATION = 10
DISTRESS_FRAMES_REQUIRED = 10
CALM_FRAMES_REQUIRED = 30
HISTORY_SIZE = 5

# =====================
# Helper Functions
# =====================
def preprocess_frame(frame, face_rect):
    """Extract face ROI and preprocess for model input"""
    x, y, w, h = face_rect
    face = frame[y:y+h, x:x+w]
    if face.size == 0:
        return np.zeros((1, 48, 48, 1), dtype=(np.uint8 if is_quantized else np.float32))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
    if is_quantized:
        return face.reshape(1, 48, 48, 1).astype(np.uint8)
    else:
        return (face / 255.0).reshape(1, 48, 48, 1).astype(np.float32)

def dequantize_output(val):
    if is_quantized:
        return (val.astype(np.float32) - output_zero_point) * output_scale
    return val

# =====================
# Webcam Setup
# =====================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("✗ Error: Webcam not opened.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
print("✓ Webcam initialized successfully.")

window_name = "Wellness Monitor - Emotion Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

# =====================
# State Variables
# =====================
current_song = None
music_playing = False
music_start_time = 0
last_distress_time = 0
last_calm_time = time.time()
frame_count = 0
distress_streak = 0
calm_streak = 0
emotion_history = []
verbose = False
fullscreen = False
current_emotion = "Unknown"
emotion_confidence = 0.0

# =====================
# User Instructions
# =====================
print("\n" + "="*50)
print("CONTROLS:")
print("="*50)
print("Q - Quit")
print("P - Pause/Resume music")
print("T - Toggle threshold (0.5 ↔ 0.7)")
print("V - Toggle verbose output")
print("F - Toggle fullscreen")
print("="*50 + "\n")

# =====================
# Main Loop
# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    distress_detected = False
    face_found = False

    for (x, y, w, h) in faces:
        face_found = True
        face_img = preprocess_frame(frame, (x, y, w, h))
        try:
            interpreter.set_tensor(input_details[0]['index'], face_img)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            prediction_value = dequantize_output(output[0][0])
            distress_detected = prediction_value > THRESHOLD

            current_emotion = "Distressed/Anxious" if distress_detected else "Happy/Calm"
            emotion_confidence = prediction_value if distress_detected else 1.0 - prediction_value

            # Smooth prediction
            emotion_history.append(distress_detected)
            if len(emotion_history) > HISTORY_SIZE:
                emotion_history.pop(0)
            distress_detected = sum(emotion_history) > len(emotion_history) / 2

            if verbose:
                print(f"Frame {frame_count}: {prediction_value:.3f} → {current_emotion}")

            # Draw rectangle
            color = (0, 0, 255) if distress_detected else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        except Exception as e:
            if verbose:
                print("Inference error:", e)

    # =====================
    # Emotion Stability Logic
    # =====================
    current_time = time.time()
    if distress_detected:
        distress_streak += 1
        calm_streak = 0
    else:
        calm_streak += 1
        distress_streak = 0

    if distress_streak >= DISTRESS_FRAMES_REQUIRED and not music_playing and (current_time - last_distress_time) > COOLDOWN:
        current_song = random.choice(song_files)
        mixer.music.load(current_song)
        mixer.music.play()
        music_playing = True
        music_start_time = current_time
        last_distress_time = current_time
        print(f"♪ Playing: {os.path.basename(current_song)}")
        distress_streak = 0

    if calm_streak >= CALM_FRAMES_REQUIRED and music_playing:
        mixer.music.stop()
        music_playing = False
        calm_streak = 0
        print("✓ Calm state detected, music stopped.")

    if music_playing and (current_time - music_start_time) > PLAY_DURATION:
        mixer.music.stop()
        music_playing = False

    # =====================
    # UI Overlay
    # =====================
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = w / 1000
    thickness = 2

    status = "No Face Detected" if not face_found else ("⚠ DISTRESS DETECTED" if distress_detected else "✓ Normal State")
    color = (0, 0, 255) if "DISTRESS" in status else ((0, 255, 0) if "Normal" in status else (128, 128, 128))
    cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.putText(frame, status, (20, 50), font, font_scale * 1.2, color, thickness, cv2.LINE_AA)
    if face_found:
        cv2.putText(frame, f"Emotion: {current_emotion} | Conf: {emotion_confidence:.2f} | Thr: {THRESHOLD}",
                    (20, 100), font, font_scale, (255, 255, 255), thickness - 1, cv2.LINE_AA)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        if music_playing:
            mixer.music.stop()
            music_playing = False
            print("⏸ Music paused.")
        elif current_song and not mixer.music.get_busy():
            mixer.music.play()
            music_playing = True
            music_start_time = time.time()
            print("▶ Music resumed.")
    elif key == ord('t'):
        THRESHOLD = 0.7 if THRESHOLD == 0.5 else 0.5
        print(f"⚙ Threshold changed to {THRESHOLD}")
    elif key == ord('v'):
        verbose = not verbose
        print(f"⚙ Verbose mode: {'ON' if verbose else 'OFF'}")
    elif key == ord('f'):
        fullscreen = not fullscreen
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

# Cleanup
cap.release()
cv2.destroyAllWindows()
mixer.quit()
print("✓ Program terminated successfully.")
