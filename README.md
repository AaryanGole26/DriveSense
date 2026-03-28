# DriveSense Wellness Monitor 🚗🧠

DriveSense is an AI-powered driver wellness monitoring system that detects emotional states and signs of distress in real-time. Using Computer Vision and TensorFlow Lite, it identifies when a driver might be anxious or distressed and automatically triggers calming music to help them relax.

## ✨ Features
- **Real-time Emotion Detection**: Analyzes facial expressions to identify "Distressed/Anxious" vs. "Happy/Calm" states.
- **Automated Music Therapy**: Seamlessly plays curated tracks from your collection when distress is detected.
- **Premium HUD Overlay**: A futuristic, semi-transparent interface (Glassmorphism) for clear status updates and "Now Playing" indicators.
- **Quantized Model**: Efficient INT8 TFLite model for optimized performance on edge devices.
- **Dynamic Threshold Support**: Toggle between sensitivity levels for different environments.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Webcam
- Audio output device

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AaryanGole26/DriveSense.git
   cd DriveSense
   ```
2. Set up the virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
1. Place your MP3 tracks in the `songs/` directory.
2. Ensure you have the `distress_model.tflite` file in the root directory.
3. Keep `songs/` and `dataset/` local on your machine (they are intentionally not pushed to git).

## 🎮 Controls
| Key | Action |
|-----|--------|
| **Q** | Quit Program |
| **P** | Pause/Resume Music |
| **T** | Toggle Sensitivity (0.5 ↔ 0.7) |
| **V** | Toggle Verbose Console Logs |
| **F** | Toggle Fullscreen Mode |
| **(X)** | Close Window to Exit |

## 🛠 Project Structure
- `wellness_monitor.py`: Main application entry point.
- `train_quantized_model.py`: Training script for the TFLite model.
- `prepare_dataset.py`: Utility for dataset preprocessing.
- `songs/`: Directory containing calming tracks.
- `dataset/`: Training data source.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
