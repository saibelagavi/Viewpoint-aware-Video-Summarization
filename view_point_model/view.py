import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Constants
MODEL_PATH = '/content/convlstm_model.keras'  # Replace with the path to your trained model
IMAGE_HEIGHT, IMAGE_WIDTH = 32, 32  # Match the resolution of your model's input
SEQUENCE_LENGTH = 10  # Number of frames in a sequence
DESIRED_ACTION_CLASS = "Biking"  # The desired view(action) class
THRESHOLD = 0.4 # Action detection threshold

# Define class labels based on your model's classes
class_labels = [
    "BaseballPitch",
    "Basketball",
    "BenchPress",
    "Biking",
    "Billiards",
    "BreastStroke",
    "CleanAndJerk",
    "Diving",
    "Drumming",
    "Fencing"
]

# Load the trained model
model = load_model(MODEL_PATH)

# Open the input video file
input_video_path = '/content/drive/MyDrive/input.mp4'  # Replace with the path to your input video
cap = cv2.VideoCapture(input_video_path)

# Create an output video file with the original resolution
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = '/content/drive/MyDrive/output_video.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (IMAGE_WIDTH, IMAGE_HEIGHT))

# Variables to store frames for the sequence
frame_buffer = []

# Variable to track if the desired action is detected
action_detected = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to match the model's input size
    resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    normalized_frame = resized_frame / 255.0

    # Add the frame to the buffer
    frame_buffer.append(normalized_frame)

    # Check if the buffer has enough frames for a sequence
    if len(frame_buffer) == SEQUENCE_LENGTH:
        # Predict on the sequence
        sequence = np.array(frame_buffer)
        prediction = model.predict(np.array([sequence]))

        # Check if the model detects the desired action
        if prediction[0][class_labels.index(DESIRED_ACTION_CLASS)] > THRESHOLD:
            action_detected = True
        else:
            action_detected = False

        # If action is detected, save the frames with the action to the output video
        if action_detected:
            for saved_frame in frame_buffer:
                out.write((saved_frame * 255).astype(np.uint8))
        frame_buffer = []

cap.release()
out.release()
