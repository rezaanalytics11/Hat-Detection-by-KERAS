from keras import datasets, layers, models
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


train_dataset_path=r"C:\Users\Ariya Rayaneh\Desktop\akhond"
width=height=224
batch_size=16

idg=ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True,
    brightness_range=(0.8,1.2),
    zoom_range=0.1,
    shear_range=0.3,
    rotation_range=10,
    validation_split=0.2
)


train_data=idg.flow_from_directory(
    train_dataset_path,
    target_size=(width,height),
    class_mode='categorical',
    batch_size=batch_size,
    subset='training'
)

val_data=idg.flow_from_directory(
    train_dataset_path,
    target_size=(width, height),
    class_mode='categorical',
    batch_size=batch_size,
    subset='validation'
)

model = keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss=tf.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50
)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
#plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss,Accuracy(Train & Val')
plt.title('loss & Accuracy vs Epoch(val)')
plt.legend()
plt.savefig(r"C:\Users\Ariya Rayaneh\Desktop\akhond\fig1.png")



# y_true = np.concatenate([y for x, y in test_ds], axis=0)
# y_pred = model.predict(test_ds).flatten()
# y_pred = np.round(y_pred)
# cm = confusion_matrix(y_true, y_pred)

# sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.savefig(r"C:\Users\Ariya Rayaneh\Desktop\assignment-44\fig22.png")
# plt.show()




# test_ds = keras.preprocessing.image_dataset_from_directory(
# r"C:\Users\Ariya Rayaneh\Desktop\archive (36)",
#     image_size=(224, 224),
#     batch_size=16,
#     label_mode='binary'
# )
# y_true = np.concatenate([y for x, y in test_ds], axis=0)
# y_pred = model.predict(test_ds).flatten()
# y_pred = np.round(y_pred)
# accuracy = np.mean(y_pred == y_true)
#
# history = model.fit(
#     train_ds,
#     validation_data=test_ds,
#     epochs=5
# )

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='test_accuracy')
# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='test_loss')
# plt.xlabel('Epoch')
# plt.ylabel('loss,Accuracy(Train & test')
# plt.title('loss & Accuracy vs Epoch(test)')
# plt.legend()
# plt.savefig(r"C:\Users\Ariya Rayaneh\Desktop\archive (36)\fig3.png")