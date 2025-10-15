import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from pygame import mixer
import time
import os
import random

# ============================================
# WELLNESS MONITOR – Emotion & Distress Detection
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

# 🧍 Initialize MediaPipe
try:
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    print("✓ MediaPipe face mesh initialized.")
except Exception as e:
    print(f"✗ Error initializing MediaPipe: {e}")
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
def preprocess_frame(frame, landmarks):
    """Extract face ROI and preprocess for model input"""
    h, w = frame.shape[:2]
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    x_min = int(min(x_coords) * w) - 20
    x_max = int(max(x_coords) * w) + 20
    y_min = int(min(y_coords) * h) - 20
    y_max = int(max(y_coords) * h) + 20
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)
    face = frame[y_min:y_max, x_min:x_max]
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
show_face_mesh = True
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
print("M - Toggle face mesh overlay")
print("="*50 + "\n")

# =====================
# Main Loop
# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    distress_detected = False
    face_found = False

    if results.multi_face_landmarks:
        face_found = True
        for face_landmarks in results.multi_face_landmarks:
            face_img = preprocess_frame(frame, face_landmarks.landmark)
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

                if show_face_mesh:
                    mp_drawing.draw_landmarks(
                        frame, face_landmarks,
                        mp_face.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
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

    # 🎵 Play music only after sustained distress
    if distress_streak >= DISTRESS_FRAMES_REQUIRED and not music_playing and (current_time - last_distress_time) > COOLDOWN:
        current_song = random.choice(song_files)
        mixer.music.load(current_song)
        mixer.music.play()
        music_playing = True
        music_start_time = current_time
        last_distress_time = current_time
        print(f"♪ Playing: {os.path.basename(current_song)}")
        distress_streak = 0

    # ⏹ Stop music only after sustained calmness
    if calm_streak >= CALM_FRAMES_REQUIRED and music_playing:
        mixer.music.stop()
        music_playing = False
        calm_streak = 0
        print("✓ Calm state detected, music stopped.")

    # Stop song after timeout
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

    # =====================
    # Key Controls
    # =====================
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
    elif key == ord('m'):
        show_face_mesh = not show_face_mesh
        print(f"⚙ Face mesh: {'ON' if show_face_mesh else 'OFF'}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
mixer.quit()
print("✓ Program terminated successfully.")
