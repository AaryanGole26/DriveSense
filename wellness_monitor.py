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
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector is None or face_detector.empty():
        # Fallback to local file if it exists, or just raise
        if os.path.exists("haarcascade_frontalface_default.xml"):
             face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    if face_detector.empty():
        raise Exception(f"Failed to load Haar Cascade from {cascade_path}")
    print(f"✓ OpenCV face detector initialized from {os.path.basename(cascade_path)}")
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

def draw_hud(img, status, color, emotion, conf, threshold, music_info=None):
    """Draw a premium Glassmorphism-style HUD"""
    fh, fw = img.shape[:2]
    
    # 🌫️ Top Bar (Glass Effect)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (fw, 100), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # Status Line
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, status, (30, 50), font, 1.2, color, 2, cv2.LINE_AA)
    
    # Details Line
    details = f"Emotion: {emotion} | Confidence: {conf:.2f} | Thr: {threshold}"
    cv2.putText(img, details, (30, 85), font, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
    
    # 🏷️ DriveSense Branding
    brand_text = "DRIVESENSE AI"
    (tw, th), _ = cv2.getTextSize(brand_text, font, 0.8, 2)
    cv2.putText(img, brand_text, (fw - tw - 30, 50), font, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
    
    # 🎵 Bottom "Now Playing" Bar
    if music_info:
        # Translucent bar at bottom
        overlay = img.copy()
        cv2.rectangle(overlay, (0, fh - 70), (fw, fh), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Song Info
        music_text = f"♪ NOW PLAYING: {music_info}"
        cv2.putText(img, music_text, (30, fh - 30), font, 0.7, (0, 255, 100), 1, cv2.LINE_AA)
        
        # Simple Progress Visualizer (Pulsing)
        pulse = int(10 * np.sin(time.time() * 5)) + 15
        cv2.circle(img, (fw - 50, fh - 35), pulse, (0, 255, 100), 2)

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
    # Check if window was closed (X button)
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        print("✓ Window closed by user.")
        break

    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Fix for Task 2: Safety check and try-except for detectMultiScale
        if face_detector is not None:
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        else:
            faces = []
    except Exception as detection_error:
        if verbose:
            print(f"Detection error: {detection_error}")
        faces = []

    distress_detected = False
    face_found = bool(len(faces) > 0)

    # Use a flag for the current frame's distress status
    frame_distress = False

    for (x, y, w, h) in faces:
        face_img = preprocess_frame(frame, (x, y, w, h))
        try:
            interpreter.set_tensor(input_details[0]['index'], face_img)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            prediction_value = dequantize_output(output[0][0])
            
            # Prediction logic
            is_distressed = prediction_value > THRESHOLD
            if is_distressed:
                frame_distress = True

            current_emotion = "Distressed/Anxious" if is_distressed else "Happy/Calm"
            emotion_confidence = prediction_value if is_distressed else 1.0 - prediction_value

            if verbose:
                print(f"Frame {frame_count}: {prediction_value:.3f} → {current_emotion}")

            # Draw elegant rectangle with corners
            color = (0, 0, 255) if is_distressed else (0, 255, 0)
            length = 30
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1) # Thin box
            # Corners
            cv2.line(frame, (x, y), (x + length, y), color, 4)
            cv2.line(frame, (x, y), (x, y + length), color, 4)
            cv2.line(frame, (x + w, y), (x + w - length, y), color, 4)
            cv2.line(frame, (x + w, y), (x + w, y + length), color, 4)
            cv2.line(frame, (x, y + h), (x + length, y + h), color, 4)
            cv2.line(frame, (x, y + h), (x, y + h - length), color, 4)
            cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, 4)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, 4)
            
        except Exception as e:
            if verbose:
                print("Inference error:", e)

    # Smooth prediction based on the frame's findings
    emotion_history.append(frame_distress)
    if len(emotion_history) > HISTORY_SIZE:
        emotion_history.pop(0)
    distress_detected = sum(emotion_history) >= len(emotion_history) / 2

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
        try:
            mixer.music.load(current_song)
            mixer.music.play()
            music_playing = True
            music_start_time = current_time
            last_distress_time = current_time
            print(f"♪ Playing: {os.path.basename(current_song)}")
        except Exception as music_err:
            print(f"✗ Music play error: {music_err}")
        distress_streak = 0

    # Stop logic (Distress ended)
    if calm_streak >= CALM_FRAMES_REQUIRED and music_playing:
        mixer.music.fadeout(2000) # Smoother stop
        music_playing = False
        calm_streak = 0
        print("✓ Calm state detected, music stopped.")

    # Timeout logic
    if music_playing and (current_time - music_start_time) > PLAY_DURATION:
        mixer.music.fadeout(2000)
        music_playing = False

    # =====================
    # UI Overlay (Modified for Task 1)
    # =====================
    status = "No Face Detected" if not face_found else ("⚠ DISTRESS DETECTED" if distress_detected else "✓ Normal State")
    color = (0, 0, 255) if "DISTRESS" in status else ((0, 255, 0) if "Normal" in status else (150, 150, 150))
    
    song_name = os.path.basename(current_song) if (music_playing and current_song) else None
    
    # Use our new premium HUD
    draw_hud(frame, status, color, current_emotion if face_found else "N/A", 
             emotion_confidence if face_found else 0.0, THRESHOLD, song_name)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        if music_playing:
            mixer.music.fadeout(1000)
            music_playing = False
            print("⏸ Music paused.")
        elif current_song:
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
