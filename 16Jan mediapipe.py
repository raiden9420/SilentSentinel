import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPooling3D, Activation, \
    TimeDistributed, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras import initializers
import mediapipe as mp

# Mediapipe initialization
mp_face_mesh = mp.solutions.face_mesh
FACE_MESH = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Function to dynamically crop the lip region
def crop_lip_region(frame):
    results = FACE_MESH.process(frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract specific landmarks for lips (indices 61-88)
            lip_coords = [
                (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                for landmark in face_landmarks.landmark[61:88]
            ]
            # Calculate bounding box for lips
            x_min = max(min([coord[0] for coord in lip_coords]), 0)
            x_max = min(max([coord[0] for coord in lip_coords]), frame.shape[1])
            y_min = max(min([coord[1] for coord in lip_coords]), 0)
            y_max = min(max([coord[1] for coord in lip_coords]), frame.shape[0])
            return frame[y_min:y_max, x_min:x_max]
    return None

# Load video function with lip cropping
def load_video(path: str) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        lip_region = crop_lip_region(frame)  # Crop lip region dynamically
        if lip_region is not None:
            lip_region = cv2.resize(lip_region, (140, 46))  # Resize to consistent size
            lip_region = cv2.cvtColor(lip_region, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
            frames.append(lip_region[..., np.newaxis])  # Add channel dimension
    cap.release()

    # Normalize frames
    frames = np.array(frames, dtype=np.float32)
    mean = np.mean(frames)
    std = np.std(frames)
    return (frames - mean) / std

# Load alignments (unchanged from original)
def load_alignments(path: str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

# Load data for TensorFlow dataset
def load_data(path: str):
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    video_path = os.path.join('/kaggle/input/newdata/newdata/s1', f'{file_name}.mpg')
    alignment_path = os.path.join('/kaggle/input/newdata/newdata/alignment/s1', f'{file_name}.align')
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)
    return frames, alignments

def mappable_function(path: str) -> List[str]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result

# Define the vocabulary and create StringLookup layers
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# Prepare dataset
data = tf.data.Dataset.list_files('/kaggle/input/newdata/newdata/s1/*.mpg')
data = data.shuffle(2500, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(8, padded_shapes=([75, 46, 140, 1], [40]))
data = data.cache().prefetch(tf.data.AUTOTUNE)

train = data.take(2000)
test = data.skip(2000)

# Build the model
model = Sequential()
model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling3D((1, 2, 2)))
model.add(BatchNormalization())

model.add(Conv3D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling3D((1, 2, 2)))
model.add(BatchNormalization())

model.add(Conv3D(75, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling3D((1, 2, 2)))
model.add(BatchNormalization())

model.add(TimeDistributed(Flatten()))

model.add(Bidirectional(LSTM(128, kernel_initializer=initializers.Orthogonal(), return_sequences=True)))
model.add(Dropout(0.3))

model.add(Bidirectional(LSTM(128, kernel_initializer=initializers.Orthogonal(), return_sequences=True)))
model.add(Dropout(0.3))

model.add(Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'))

# Define CTC Loss
def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=15, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train model
model.fit(
    train,
    validation_data=test,
    epochs=100,
    steps_per_epoch=2000 // 8,
    validation_steps=500 // 8,
    callbacks=[reduce_lr, early_stopping],
    verbose=1
)
