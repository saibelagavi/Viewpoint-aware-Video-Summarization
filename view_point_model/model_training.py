import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 32, 32  # Reduced image size
SEQUENCE_LENGTH = 10  # Reduced sequence length

# Specify the dataset directory path in Google Colab.
# Use the path where you uploaded your dataset.
DATASET_DIR = '/content/drive/MyDrive/UCF50'  # Update this path

# Define the list of class labels
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

# Automatically get the number of classes
NUM_CLASSES = len(class_labels)

# Functions to extract frames and create the dataset
def frames_extraction(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()

        if not success:
            break

        # Downsample the frame to a smaller size
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)

    video_reader.release()
    return frames_list

def one_hot_encode(label):
    label_index = class_labels.index(label)
    one_hot = [0] * NUM_CLASSES
    one_hot[label_index] = 1
    return one_hot

def create_dataset():
    features = []
    labels = []

    for class_name in class_labels:
        print(f'Extracting Data of Class: {class_name}')
        class_dir = os.path.join(DATASET_DIR, class_name)
        files_list = os.listdir(class_dir)

        for file_name in files_list:
            video_file_path = os.path.join(class_dir, file_name)
            frames = frames_extraction(video_file_path)

            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(class_name)

    features = np.array(features)
    labels = np.array(labels)
    return features, labels

# Load your dataset, preprocess it, and split it into training and testing data
features, labels = create_dataset()

# One-hot encode the labels
one_hot_labels = [one_hot_encode(label) for label in labels]

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, one_hot_labels, test_size=0.25, random_state=27)

# Create and compile your ConvLSTM model
def create_convlstm_model():
    model = Sequential()
    model.add(ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh', recurrent_activation='sigmoid', data_format='channels_last', return_sequences=True, input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))

    # You can add more layers as needed, such as Conv2D, MaxPooling2D, Flatten, and Dense layers.

    model.add(Flatten())
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_convlstm_model()

# Train the model on your training data
NUM_EPOCHS = 30
BATCH_SIZE = 4
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

model.fit(X_train, np.array(y_train), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=[early_stopping_callback])

# Save the trained model in the native Keras format
model.save('convlstm_model.keras')

# Save the model weights to an H5 file
model.save_weights('convlstm_model_weights.h5')
