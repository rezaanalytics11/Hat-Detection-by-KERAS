import os
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

import wandb
from wandb.keras import WandbCallback

dataset_path = r"C:\Users\Ariya Rayaneh\Desktop\akhond\train"
width = height = 224
batch_size = 32

train_data_generator = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range = 0.1,
                                            width_shift_range=0.1, height_shift_range=0.1,
                                            horizontal_flip=True, brightness_range=(0.9, 1.1),
                                            fill_mode='reflect',validation_split=0.2
                                            )

train_data = train_data_generator.flow_from_directory((dataset_path),
                                                        target_size=(width, height),
                                                        class_mode='categorical',
                                                        batch_size=batch_size,
                                                        shuffle=True
                                                        )
val_data_generator = ImageDataGenerator(rescale=1./255)

val_data = val_data_generator.flow_from_directory((dataset_path),
                                                    target_size=(width, height),
                                                    class_mode='categorical',
                                                    batch_size=batch_size,
                                                    shuffle=False
                                                    )
train_images = next(train_data)[0]
plt.figure(figsize=(8,8)) 

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)


base_model = tf.keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(width, height, 3),
    pooling='avg'
)
for layer in base_model.layers:
    layer.trainable = False

base_model.summary()

model = tf.keras.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation='softmax')
])


model.summary()


model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
epochs = 10

history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples/batch_size,
    validation_data=val_data,
    validation_steps=val_data.samples/batch_size,
    epochs=epochs,
    shuffle=True
    #callbacks=[WandbCallback()]
)
plt.plot(history.history['loss'], label="Train")
plt.plot(history.history['val_loss'], label="Validation")
plt.legend(loc='best')
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.grid()
plt.show()

plt.plot(history.history['accuracy'], label="Train")
plt.plot(history.history['val_accuracy'], label="Validation")
plt.legend(loc='best')
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.grid()
plt.show()

