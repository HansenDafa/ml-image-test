import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

class CIFAR10Model:
    def __init__(self):
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.model = self._create_model()
        self._load_weights()

    def _create_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))
        return model

    def _load_weights(self):
        (train_images, train_labels), _ = datasets.cifar10.load_data()
        train_images = train_images / 255.0
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model.fit(train_images, train_labels, epochs=10)
    
    def predict(self, img_array):
        prediction = self.model.predict(img_array)
        predicted_class = self.class_names[np.argmax(prediction)]
        return predicted_class

# Inisialisasi model saat modul dimuat
cifar10_model = CIFAR10Model()
