# 

import tensorflow as tf
from tensorboard.plugins import projector
import numpy as np

# Define the data for the three models
resnet = [90.248, 93.0348,92.039,93.432]
vgg16 = [90.54,97.80,97.94,98.59]
vit = [97.51,97.76,98.08,98.62]
training_size = [25, 50, 75, 100]

# Create a summary writer for TensorBoard
writer = tf.summary.create_file_writer('logs')

# Create a step variable for the x-axis
step = 0

# Loop through the data and write each value to TensorBoard
for r, v, vi, t in zip(resnet, vgg16, vit, training_size):
    with writer.as_default():
        # Write the accuracy for each model
        tf.summary.scalar('ResNet', r, step=step)
        tf.summary.scalar('VGG16', v, step=step)
        tf.summary.scalar('ViT', vi, step=step)
        # Write the training data size for each iteration
        tf.summary.scalar('Training Size', t, step=step)
        
    step += 1

# Close the summary writer
writer.close()

# Write the accuracy for each model with smoothing
tf.summary.scalar('ResNet', r, step=step, smoothing_exponent=0.6)
tf.summary.scalar('VGG16', v, step=step, smoothing_exponent=0.6)
tf.summary.scalar('ViT', vi, step=step, smoothing_exponent=0.6)
