# -*- coding: utf-8 -*-
"""
#S3_V4

# Imports
"""

# from google.colab import drive
# drive.mount('/content/drive')

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Lambda
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
import ipywidgets as widgets
from tensorflow.keras.metrics import Precision, Recall

!pip install visualkeras

"""# Data pre-processing

Change the dataset Directory
"""

dataset_directory = "/content/drive/MyDrive/UAIC/FishPak" # @param {type:"string"}

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 6

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_directory,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size = None
)

class_names = dataset.class_names

def get_dataset_partitions_tf(ds, train_split=0.7, val_split=0.1, test_split=0.2, shuffle=True, shuffle_size=30):

    ds_size = len(ds)

    if shuffle:
        ds.shuffle(shuffle_size, seed=12)

    train_size = int(ds_size * train_split)
    val_size = int(ds_size * val_split)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 6

train_ds = train_ds.cache().shuffle(1000).batch(BATCH_SIZE, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).batch(BATCH_SIZE, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).batch(BATCH_SIZE, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds

test_ds

scaled_images_list = []
for images, _ in train_ds:
    scaled_images = tf.constant(images.numpy() / 255.0)
    scaled_images_list.append(scaled_images)

train_ds = tf.data.Dataset.from_tensor_slices(scaled_images_list)

"""# Set Path and Create Model
*   Set to None if you donot have a pretrained model and want to create one
*   Else Enter the path to the model (.h5) in the field provided


"""

model_path = "None" # @param ["None"] {allow-input: true}

from tensorflow.keras import backend as K

def f1_score(y_true, y_pred):
    precision = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / (K.sum(K.round(K.clip(y_pred, 0, 1))) + K.epsilon())
    recall = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / (K.sum(K.round(K.clip(y_true, 0, 1))) + K.epsilon())
    return 2 * (precision * recall) / (precision + recall + K.epsilon())

def create_custom_lightweight_model(input_shape, n_classes):
    # Input Layer
    inputs = layers.Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3))

    # Initial Convolution Block
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    # Depthwise Separable Convolution Block
    def depthwise_separable_conv_block(x, filters, strides=(1, 1)):
        # First Depthwise Separable Conv Layer
        x = layers.DepthwiseConv2D((3, 3), strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, (1, 1), padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Second Depthwise Separable Conv Layer
        x = layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, (1, 1), padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU()(x)

    # Stacking depthwise separable conv blocks
    x = depthwise_separable_conv_block(x, 64)
    x = depthwise_separable_conv_block(x, 128, strides=(2, 2))
    x = depthwise_separable_conv_block(x, 128)
    x = depthwise_separable_conv_block(x, 256, strides=(2, 2))
    x = depthwise_separable_conv_block(x, 256)
    x = depthwise_separable_conv_block(x, 512, strides=(2, 2))

    # Global Average Pooling and Output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inputs, x)

    return model

input_shape

!pip install git+https://github.com/paulgavrikov/visualkeras --upgrade
import visualkeras

model = create_custom_lightweight_model(input_shape, n_classes)

visualkeras.layered_view(model, legend=True, draw_volume=False)

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

# Convert the Keras model to a TensorFlow function
concrete_func = tf.function(lambda x: model(x))
concrete_func = concrete_func.get_concrete_function(
    tf.TensorSpec([1] + list((256,256,3)), model.input.dtype)
)

# Freeze the function and convert to constants
frozen_func = convert_variables_to_constants_v2(concrete_func)

# Profile the model to calculate FLOPs
graph_info = profile(
    graph=frozen_func.graph,
    options=ProfileOptionBuilder.float_operation()
)

print(f"FLOPs: {graph_info.total_float_ops}")

"""# Loss
Contrastive Loss Training
"""

class CustomAugment(object):
    def __call__(self, sample):
        # Random crops
        h = np.random.uniform(int(0.08*tf.cast(tf.shape(sample)[1], tf.float32).numpy()), tf.shape(sample)[1].numpy(), 1)
        w = np.random.uniform(int(0.08*tf.cast(tf.shape(sample)[2], tf.float32).numpy()), tf.shape(sample)[2].numpy(), 1)
        sample = tf.image.random_crop(sample, [1, int(h), int(w), 3])
        sample = tf.image.resize(sample, size=[IMAGE_SIZE, IMAGE_SIZE])

        # Random flips
        sample = self._random_apply(tf.image.flip_left_right, sample, p=0.5)

        # randomly apply transformation (color distortions and blur) with probability p.
        sample = self._random_apply(self._color_jitter, sample, p=0.8)
        sample = self._random_apply(self._color_drop, sample, p=0.2)
        sample = self._random_apply(self._gaussian_blur, sample, p=0.5)

        return sample

    def _color_jitter(self, x, s=0.5):
        # one can also shuffle the order of following augmentations
        # each time they are applied.
        x = tf.image.random_brightness(x, max_delta=0.8*s)
        x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_hue(x, max_delta=0.2*s)
        x = tf.clip_by_value(x, 0, 1)
        return x

    def _color_drop(self, x):
        image = tf.image.rgb_to_grayscale(x)
        image = tf.tile(x, [1, 1, 1, 3])
        return x

    def _gaussian_blur(self, image, sigma=tf.random.uniform([], 0.1, 2.0, dtype=tf.float32),
                       padding='SAME'):
        kernel_size = image.shape[1]
        radius = tf.cast(kernel_size / 2, dtype=tf.int32)
        kernel_size = radius * 2 + 1

        x = tf.cast(tf.range(-radius, radius + 1), tf.float32)
        blur_filter = tf.exp(
          -tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(sigma, tf.float32), 2.0)))
        blur_filter /= tf.reduce_sum(blur_filter)

        # One vertical and one horizontal filter.
        blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
        blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
        num_channels = tf.shape(image)[-1]
        blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
        blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])

        expand_batch_dim = image.shape.ndims == 3
        if expand_batch_dim:
            # Tensorflow requires batched input to convolutions, which we can fake with
            # an extra dimension.
            image = tf.expand_dims(image, axis=0)

        blurred = tf.nn.depthwise_conv2d(
            image, blur_h, strides=[1, 1, 1, 1], padding=padding)
        blurred = tf.nn.depthwise_conv2d(
            blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)

        if expand_batch_dim:
            blurred = tf.squeeze(blurred, axis=0)

        blurred = tf.clip_by_value(blurred, 0., 1.)
        return blurred

    def _random_apply(self, func, x, p):
        return tf.cond(
          tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                  tf.cast(p, tf.float32)),
          lambda: func(x),
          lambda: x)

LARGE_NUM = 1e9

def nt_xentloss(hidden_1,
		hidden_2,
		hidden_norm=True,
		temperature=1.0,
		weights=1.0):

	# Get (normalized) hidden1 and hidden2.
	if hidden_norm:
		hidden_1 = tf.math.l2_normalize(hidden_1, -1)
		hidden_2 = tf.math.l2_normalize(hidden_2, -1)
	# hidden1, hidden2 = tf.split(hidden, 2, 0)
	batch_size = tf.shape(hidden_1)[0]

	hidden1_large = hidden_1
	hidden2_large = hidden_2
	labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
	masks = tf.one_hot(tf.range(batch_size), batch_size)

	logits_aa = tf.matmul(hidden_1, hidden1_large, transpose_b=True) / temperature
	logits_aa = logits_aa - masks
	logits_bb = tf.matmul(hidden_2, hidden2_large, transpose_b=True) / temperature
	logits_bb = logits_bb - masks
	logits_ab = tf.matmul(hidden_1, hidden2_large, transpose_b=True) / temperature
	logits_ba = tf.matmul(hidden_2, hidden1_large, transpose_b=True) / temperature

	loss_a = tf.compat.v1.losses.softmax_cross_entropy(
		labels, tf.concat([logits_ab, logits_aa], 1), weights=weights)
	loss_b = tf.compat.v1.losses.softmax_cross_entropy(
		labels, tf.concat([logits_ba, logits_bb], 1), weights=weights)
	loss = loss_a + loss_b

	return loss, logits_ab, labels

EPOCHS = 200

data_augmentation = Sequential([
    Lambda(CustomAugment())
])

def train_model_1(optimizer, model, temperature):
    step_wise_loss = []
    epoch_wise_loss = []

    for epoch in range(EPOCHS):
        for image_batch in train_ds:
            # print(image_batch)
            a = data_augmentation(image_batch)
            b = data_augmentation(image_batch)
            # print(a)
            with tf.GradientTape() as tape:
                representation_a = model(a)
                representation_b = model(b)

                loss_value, _, _ = nt_xentloss(representation_a, representation_b, temperature=temperature)


            gradients = tape.gradient(loss_value, model.trainable_weights)

            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

            step_wise_loss.append(loss_value.numpy())

        epoch_wise_loss.append(np.mean(step_wise_loss))

        print("epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))

    return epoch_wise_loss, model

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
epoch_wise_loss, model  = train_model_1(optimizer, model, temperature=0.1)

plt.plot(epoch_wise_loss)
plt.title("tau = 0.1, h1 = 512, h2 = 256")
plt.show()

import datetime

# Get current date and time
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Generate file name with current date and time
file_name = f"model_{current_time}.h5"

# Save the model with the generated file name
model.save("/content/drive/MyDrive/UAIC/models/" + file_name)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

"""# Tests"""

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

train_ds = train_ds.cache().shuffle(1000).batch(BATCH_SIZE, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).batch(BATCH_SIZE, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).batch(BATCH_SIZE, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)

model.summary()

test_ds

# Train the classification model
epochs = 100
model_fit = model
history = model_fit.fit(train_ds, epochs=epochs,validation_data=val_ds)

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Test the classification model
test_results = model_fit.evaluate(test_ds)

# Extract test accuracy from test results
test_accuracy = test_results[1]

# Print the test accuracy
print("Test Accuracy:", test_accuracy)