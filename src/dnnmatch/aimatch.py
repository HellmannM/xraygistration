# aimatch.py

import ctypes as c
import matplotlib.pyplot as plotter_lib
import numpy as np
import PIL as image_lib
import random as r
import sys
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Renderer setup -----------------------------------------------------
# Load C++ rendering libs
stdc      = c.cdll.LoadLibrary("libc.so.6")
stdcpp    = c.cdll.LoadLibrary("libc++.so.1")
renderlib = c.cdll.LoadLibrary("./src/match/librender.so")
# Create and init renderer instance
create_renderer_wrapper = renderlib.create_renderer
create_renderer_wrapper.restype = c.c_void_p
renderer = c.c_void_p(create_renderer_wrapper())
arg_buffers = [c.create_string_buffer(b"./src/match/match"),
               c.create_string_buffer(b"../testfiles/Dummy_Paul_nifti/2__head_10_stx_head.nii"),
               c.create_string_buffer(b"-device"),
               c.create_string_buffer(b"gpu")
              ]
arg_ptrs    = (c.c_char_p*4)(*map(c.addressof, arg_buffers))
renderlib.init_renderer(renderer, 4, arg_ptrs)
get_width_wrapper = renderlib.get_width
get_height_wrapper = renderlib.get_height
get_bpp_wrapper = renderlib.get_bpp
get_width_wrapper.restype = c.c_int
get_height_wrapper.restype = c.c_int
get_bpp_wrapper.restype = c.c_int
renderer_width = c.c_int(get_width_wrapper(renderer))
renderer_height = c.c_int(get_height_wrapper(renderer))
renderer_bpp = c.c_int(get_bpp_wrapper(renderer))

def get_frame(camera):
    image_buff = np.empty(shape=(renderer_width.value, renderer_height.value, renderer_bpp.value), dtype=np.uint8)
    image_buff_ptr = image_buff.ctypes.data_as(c.POINTER(c.c_uint8))
    eye_x =    (c.c_float)(camera[0])
    eye_y =    (c.c_float)(camera[1])
    eye_z =    (c.c_float)(camera[2])
    center_x = (c.c_float)(camera[3])
    center_y = (c.c_float)(camera[4])
    center_z = (c.c_float)(camera[5])
    up_x =     (c.c_float)(camera[6])
    up_y =     (c.c_float)(camera[7])
    up_z =     (c.c_float)(camera[8])
    renderlib.single_shot(renderer, image_buff_ptr, eye_x, eye_y, eye_z, center_x, center_y, center_z, up_x, up_y, up_z)
    #return image_buff
    return image_buff[:, :, 0:3]
# --------------------------------------------------------------------

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
            #TODO get volume dims.
            volume_dims = [500, 500, 500]
            volume_center = [0, 0, 0]
            eye_search_dist = [1500, 1500, 1500]
            center_search_dist = [50, 50, 50]

            eye = [
                volume_dims[0] + r.random() * eye_search_dist[0],
                volume_dims[1] + r.random() * eye_search_dist[1],
                volume_dims[2] + r.random() * eye_search_dist[2]
                ]
            center = [
                volume_center[0] + r.random() * center_search_dist[0],
                volume_center[1] + r.random() * center_search_dist[1],
                volume_center[2] + r.random() * center_search_dist[2]
                ]
            up = [r.random(), r.random(), r.random()]
            if up == [0.0, 0.0, 0.0]:
                up = [0.0, 1.0, 0.0]
            # dir and up need to be orthogonal
            cam_dir = [center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]]
            orthogonal = np.cross(cam_dir, up)
            up = np.cross(cam_dir, orthogonal)
            up = up / np.linalg.norm(up)

            y[i] = [eye[0], eye[1], eye[2], center[0], center[1], center[2], up[0], up[1], up[2]]
            X[i,] = get_frame(y[i])
        
        X = tf.keras.applications.resnet_v2.preprocess_input(X, data_format='channels_last')

        return X, y


# Parameters
dim_x = renderer_width.value
dim_y = renderer_height.value
params = {'dim': (dim_x, dim_y),
          'batch_size': 64,
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

renderlib.destroy_renderer(renderer)
