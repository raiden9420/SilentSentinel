import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPooling3D, Activation, \
    TimeDistributed, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import initializers


def load_video(path: str) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


def load_alignments(path: str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]


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


# Define the vocabulary
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# Create StringLookup layers for character to number and vice versa
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

from tensorflow.keras.initializers import Orthogonal

data = tf.data.Dataset.list_files('/kaggle/input/newdata/newdata/s1/*.mpg')
data = data.shuffle(2500, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(8, padded_shapes=([75, 46, 140, 1], [40]))

data = data.cache()
data = data.prefetch(tf.data.AUTOTUNE)
#data = data.repeat()
train = data.take(2000)
test = data.skip(2000)#.take(500)

model = Sequential()

# First Conv3D block
model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling3D((1, 2, 2)))
model.add(BatchNormalization())  # Batch Normalization after Conv3D

# Second Conv3D block
model.add(Conv3D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling3D((1, 2, 2)))
model.add(BatchNormalization())  # Batch Normalization after Conv3D

# Third Conv3D block
model.add(Conv3D(75, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling3D((1, 2, 2)))
model.add(BatchNormalization())  # Batch Normalization after Conv3D

# TimeDistributed Flatten layer
model.add(TimeDistributed(Flatten()))

# First Bidirectional LSTM
model.add(Bidirectional(LSTM(128, kernel_initializer=initializers.Orthogonal(), return_sequences=True)))
model.add(Dropout(0.3))

# Second Bidirectional LSTM
model.add(Bidirectional(LSTM(128, kernel_initializer=initializers.Orthogonal(), return_sequences=True)))
model.add(Dropout(0.3))

# Output Dense Layer with softmax activation for classification (adjust to your task)
model.add(Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'))


# def scheduler(epoch, lr):
#     if epoch < 15:
#         return lr
#     else:
#         # Convert learning rate to a float if it is a tf.Tensor
#         return float(lr * tf.math.exp(-0.1))

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=15, min_lr=1e-6)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True
)

model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)

# schedule_callback = LearningRateScheduler(scheduler)


model.fit(
    train,
    validation_data=test,
    epochs=100,
    steps_per_epoch=2000 // 8,
    validation_steps=500 // 8,
    callbacks=[reduce_lr, early_stopping],
    verbose=1
)
