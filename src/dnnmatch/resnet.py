# resnet.py

import sys
import matplotlib.pyplot as plotter_lib
import numpy as np
import PIL as image_lib
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dim=(32,32), batch_size=128, batches_per_epoch=128, n_channels=3):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.n_channels = n_channels

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        X, y = self.__data_generation()
        return X, y

    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        #X = np.empty((self.batch_size, self.n_channels, *self.dim))
        y = np.empty((self.batch_size, 9), dtype=np.float32)

        # Generate data
        for i in range(self.batch_size):
            y[i] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            #TODO X[i,] = get_frame(y[i])
            X[i,] = np.zeros((self.dim[0], self.dim[1], self.n_channels))
        
        X = tf.keras.applications.resnet_v2.preprocess_input(X, data_format='channels_last')

        return X, y


# Parameters
#TODO get dims
dim_x = 32
dim_y = 32
params = {'dim': (dim_x, dim_y),
          'batch_size': 8,
          'n_channels': 3}  #n_channels must be 3 for resnet

# Setup model
model = Sequential()
# add pretrained resnet model and make layers non-trainable
pretrained_model = tf.keras.applications.ResNet50V2(include_top=False,
                   input_shape=(dim_x, dim_y, 3),
                   pooling='avg',
                   weights='imagenet')
for each_layer in pretrained_model.layers:
        each_layer.trainable=False
model.add(pretrained_model)
# add a fully connected output layer
model.add(Flatten())
model.add(Dense(512, activation='relu'))
# 3 vec3: eye, dir, up
model.add(Dense(9, activation='linear'))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])
model.summary()

# Generators
training_generator   = DataGenerator(batches_per_epoch=128, **params)
validation_generator = DataGenerator(batches_per_epoch=32, **params)

# Train model on dataset
epochs=8
history = model.fit(x=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    use_multiprocessing=False,
                    workers=1)

# evaluate
plotter_lib.figure(figsize=(8, 8))
epochs_range = range(epochs)
plotter_lib.plot( epochs_range, history.history['mean_squared_error'], label="Training Accuracy")
plotter_lib.plot(epochs_range, history.history['val_mean_squared_error'], label="Validation Accuracy")
plotter_lib.axis(ymin=0, ymax=np.max(history.history['val_mean_squared_error']))
plotter_lib.grid()
plotter_lib.title('Model Accuracy')
plotter_lib.ylabel('Accuracy')
plotter_lib.xlabel('Epochs')
plotter_lib.legend(['train', 'validation'])
plotter_lib.show()


## make prediction
#image_pred=resnet_model.predict(image)
#image_output_class=class_names[np.argmax(image_pred)]
#print("The predicted class is", image_output_class)

## save model
#import onnxmltools
#onnx_model = onnxmltools.convert_keras(model)
#onnxmltools.utils.save_model(onnx_model, 'trained_model.onnx')
