# Sign Language Recognition Agent

A real-time sign language recognition system using MediaPipe and LSTM neural networks to detect and classify hand gestures and body poses.

## Project Overview

This project uses computer vision and deep learning to recognize sign language gestures in real-time from a webcam feed. It detects and tracks facial landmarks, body poses, and hand positions using MediaPipe's Holistic model, then uses an LSTM-based neural network to classify the gestures into predefined sign language actions.

## Features

- **Real-time Gesture Detection**: Captures video from webcam and detects hand, body, and facial landmarks
- **Pose Estimation**: Uses MediaPipe Holistic for comprehensive body pose tracking
- **Keypoint Extraction**: Extracts coordinates and visibility scores from MediaPipe results
- **LSTM Classification**: Trains an LSTM neural network to recognize sign language gestures
- **Real-time Prediction**: Displays live predictions with confidence scores
- **Visual Feedback**: Shows probability bars and recognized gestures on video feed

## Actions Recognized

- hello
- thanks
- iloveyou

## Requirements

See `requirements.txt` for complete dependency list. Key dependencies:
- Python 3.8.10
- OpenCV (cv2)
- MediaPipe
- NumPy
- TensorFlow/Keras
- scikit-learn
- Matplotlib

## Installation

1. Create a virtual environment:
```bash
python -m venv max_env
```

2. Activate the virtual environment:
   - On Windows: `max_env\Scripts\activate`
   - On macOS/Linux: `source max_env/bin/activate`

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Jupyter notebook to execute different stages of the pipeline:

```bash
jupyter notebook agent.ipynb
```

### Key Steps:

1. **Collect Training Data**: Capture video sequences for each gesture
2. **Extract Keypoints**: Convert video frames to numerical keypoint arrays
3. **Train Model**: Build and train LSTM network
4. **Evaluate**: Test model accuracy and confusion matrix
5. **Real-time Prediction**: Run live recognition on webcam feed

## Project Structure

- `agent.ipynb` - Main Jupyter notebook containing the complete pipeline
- `DATASET/` - Directory structure for storing training data sequences
- `Logs/` - TensorBoard logs directory
- `action.h5` - Saved trained model weights

## Workflow

### 1. Data Collection
The system captures 30 sequences of 30 frames each for each action. Video frames are processed through MediaPipe to extract keypoints.

### 2. Keypoint Extraction
For each frame, the system extracts:
- **Pose**: 33 keypoints (x, y, z, visibility) = 132 features
- **Face**: 468 keypoints (x, y, z) = 1404 features
- **Left Hand**: 21 keypoints (x, y, z) = 63 features
- **Right Hand**: 21 keypoints (x, y, z) = 63 features
- **Total**: 1662 features per frame

### 3. Model Architecture
```
LSTM(64) → LSTM(128) → LSTM(64) → Dense(64) → Dense(32) → Dense(3)
Input: (30, 1662)  Output: (3,) - probability for each action
```

### 4. Training
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Categorical Accuracy
- Epochs: 2000
- Train/Test Split: 95/5

### 5. Real-time Inference
- Captures continuous video feed
- Maintains 30-frame sliding window
- Makes predictions every 30 frames
- Displays confidence scores and recognized gestures

## Controls

- Press 'Q' to quit the application during real-time prediction

## Model Performance

The model is evaluated using:
- Confusion Matrix
- Accuracy Score
- Real-time prediction confidence threshold (default: 0.4)

## Files Generated

- `action.h5` - Trained model weights
- `Logs/` - TensorBoard event files for monitoring training

## Future Improvements

- Add more sign language actions
- Implement gesture smoothing for better real-time performance
- Add confidence threshold adjustments
- Create a GUI interface for easier interaction
- Optimize model for faster inference

## Notes

- Requires a working webcam for data collection and real-time prediction
- Better lighting and clear hand/body visibility improves accuracy
- Collecting diverse training data improves model generalization
- Use TensorBoard for monitoring training progress: `tensorboard --logdir=Logs`

## Author

Created as a sign language recognition prototype using MediaPipe and TensorFlow.
Nicholas Renotte a youruber.

## License

This project is provided as-is for educational and research purposes.
