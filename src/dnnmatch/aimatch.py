# aimatch.py

import ctypes as c
import matplotlib.pyplot as plotter_lib
import numpy as np
import os.path
import PIL as image_lib
import random as r
import sys
import tensorflow as tf

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
    image_buff = np.empty(shape=(renderer_height.value, renderer_width.value, renderer_bpp.value), dtype=np.uint8)
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
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.uint8)
        #y = np.empty((self.batch_size, 9), dtype=np.float32)
        y = np.empty((self.batch_size, 1), dtype=np.float32)

        # Generate data
        for i in range(self.batch_size):
            #TODO get volume dims.
            volume_dims = [500, 500, 500]
            volume_center = [0, 0, 0]
            eye_search_dist = [100, 100, 100]
            #eye_search_dist = [1500, 1500, 1500]
            center_search_dist = [10, 10, 10]

            #TODO make sure eye is outside of volume_dims
            eye = [
                1000 + r.random() * eye_search_dist[0] - eye_search_dist[0]/2,
                   0 + r.random() * eye_search_dist[1] - eye_search_dist[1]/2,
                   0 + r.random() * eye_search_dist[2] - eye_search_dist[2]/2
                ]
            center = [
                volume_center[0] + r.random() * center_search_dist[0] - center_search_dist[0]/2,
                volume_center[1] + r.random() * center_search_dist[1] - center_search_dist[1]/2,
                volume_center[2] + r.random() * center_search_dist[2] - center_search_dist[2]/2
                ]
#            up = [r.random(), r.random(), r.random()]
#            if up == [0.0, 0.0, 0.0]:
#                up = [0.0, 1.0, 0.0]
#            # dir and up need to be orthogonal
#            cam_dir = [center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]]
#            orthogonal = np.cross(cam_dir, up)
#            up = np.cross(cam_dir, orthogonal)
#            up = up / np.linalg.norm(up)
            up = [0, 1, 0]
#            eye = [1000, 0, 0]
#            center = [0, 0, 0]

            #y[i] = [eye[0], eye[1], eye[2], center[0], center[1], center[2], up[0], up[1], up[2]]
            #X[i,] = get_frame(y[i])
            y[i] = [eye[0]]
            X[i,] = get_frame([y[i], 0, 0, 0, 0, 0, 0, 1, 0])
        
        X = tf.keras.applications.resnet_v2.preprocess_input(X, data_format='channels_last')

        return X, y


# Parameters
dim_y = renderer_width.value
dim_x = renderer_height.value
params = {'dim': (dim_x, dim_y),
          'batch_size': 1,
          'n_channels': 3}  #n_channels must be 3 for resnet

# Setup model
model = tf.keras.models.Sequential()
# load from disk if present
if os.path.isfile('trained_model.keras'):
    print('loading trained model...')
    model = tf.keras.models.load_model('trained_model.keras')
else:
    print('creating new model...')
    model.add(tf.keras.layers.Resizing(height=224, width=224, interpolation="bilinear", crop_to_aspect_ratio=True))
    # add pretrained resnet model and make layers non-trainable
    resnet = tf.keras.applications.ResNet50V2(include_top=False,
                       #input_shape=(dim_x, dim_y, 3),
                       input_shape=(224, 224, 3),
                       pooling="None",
                       weights="imagenet")
    for layer in resnet.layers:
        layer.trainable=False
    model.add(resnet)
    #pretrained_model = tf.keras.models.Model(resnet.input, resnet.layers[-3].output)
    #model.add(pretrained_model)

    model.add(tf.keras.layers.AveragePooling2D((3,3)))
    model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.Dense(8192, activation='relu'))
    # 3 vec3: eye, dir, up
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])
    model.build(input_shape=(None, dim_x, dim_y, 3))
    model.summary()

# Generators
training_generator   = DataGenerator(batches_per_epoch=1, **params)
validation_generator = DataGenerator(batches_per_epoch=1, **params)

# Train model on dataset
epochs=500
print("start training...")
fit_history = model.fit(x=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    shuffle=False,
                    use_multiprocessing=False,
                    workers=1)

### evaluate
#fig, ax = plotter_lib.subplots()
#ax.plot(range(epochs), fit_history.history['mean_squared_error'], label='Training MSE')
#ax.plot(range(epochs), fit_history.history['val_mean_squared_error'], label='Validation MSE')
#ax.set(xlabel='Epochs', ylabel='Accuracy', title='Model Accuracy')
#ax.legend()
#fig.savefig('accuracy.png')
##plotter_lib.show()


## make prediction
print("predict...")
eye = [1000, 0, 0]
center = [0, 0, 0]
up = [0, 1, 0]
test_cam = [eye[0], eye[1], eye[2], center[0], center[1], center[2], up[0], up[1], up[2]]
test_image = get_frame(test_cam)
test_prediction=model.predict(np.expand_dims(test_image, axis=0))
print("Test: eye=", eye, " center=", center, " up=", up)
#print("Pred: eye=", test_prediction[0, 0:3], " center=", test_prediction[0, 3:6], " up=", test_prediction[0, 6:9], "\n")
print("Pred: eye.x=", test_prediction[0, 0], "\n")

## save model
#model.save("trained_model.keras")

renderlib.destroy_renderer(renderer)
