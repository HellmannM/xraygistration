# resnet.py

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
    def __init__(self, dim=(32,32,32), batch_size=128, batches_per_epoch=128, n_channels=1):
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
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = np.empty((self.batch_size, self.n_channels, *self.dim))
        y = np.empty((self.batch_size, 9), dtype=np.float32)

        # Generate data
        for i in range(self.batch_size):
            y = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
            #TODO X[i,] = get_frame(y)
            X[i,0,] = [[i, 1], [2, 3]]

        return X, y


# import ResNet-50 model from keras library
# Setting include_top to False means it will allow adding input and output layers custom to a problem.
# The weights parameter specifies that the model uses its weights while training on the imagenet dataset.
model = Sequential()
pretrained_model_for_demo= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(180,180,3),
                   pooling='avg',
                   weights='imagenet')
for each_layer in pretrained_model_for_demo.layers:
        each_layer.trainable=False
model.add(pretrained_model_for_demo)

# add a fully connected output layer
model.add(Flatten())
model.add(Dense(512, activation='relu'))
# 3 vec3: eye, dir, up
model.add(Dense(9, activation='linear'))


#keras.losses.MeanSquaredError(reduction="sum_over_batch_size", name="mean_squared_error")
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['accuracy'])

# Parameters
#TODO get dims
params = {'dim': (2,2,2),
          'batch_size': 128,
          'n_channels': 1}

# Generators
training_generator   = DataGenerator(batches_per_epoch=1024, **params)
validation_generator = DataGenerator(batches_per_epoch=128, **params)

# Train model on dataset
epochs=10
history = model.fit(x=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    use_multiprocessing=False,
                    workers=1)

# evaluate
plotter_lib.figure(figsize=(8, 8))
epochs_range= range(epochs)
plotter_lib.plot( epochs_range, history.history['accuracy'], label="Training Accuracy")
plotter_lib.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")
plotter_lib.axis(ymin=0.4,ymax=1)
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
