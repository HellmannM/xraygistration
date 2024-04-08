# aimatch.py

import argparse
import ctypes as c
import matplotlib.pyplot as plotter_lib
import numpy as np
import os.path
import PIL as image_lib
import random as r
import sys
import tensorflow as tf

#TF_GPU_ALLOCATOR=cuda_malloc_async
#TF_FORCE_GPU_ALLOW_GROWTH=true

#gpus = tf.config.list_physical_devices('GPU')
#if gpus:
#    try:
#        # Currently, memory growth needs to be the same across GPUs
#        for gpu in gpus:
#            tf.config.experimental.set_memory_growth(gpu, True)
#        logical_gpus = tf.config.list_logical_devices('GPU')
#        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#    except RuntimeError as e:
#        # Memory growth must be set before GPUs have been initialized
#        print(e)


## Parse cmdline args -------------------------------------------------
parser = argparse.ArgumentParser(description='Train DNN on volume and/or predict X-ray cam args.')
parser.add_argument('--load', type=str, help='Load pre-trained DNN model (path to file).')
parser.add_argument('--train', type=str, help='Train DNN model on nii volume (path to file).')
parser.add_argument('--store', type=str, help='Export trained DNN model file (path to file).')
parser.add_argument('--predict', type=str, nargs='?', help='Predict cam args for dcm files (paths to files)')
parser.add_argument('--export_predictions', type=str, help='Export predicted coords as json (path to file).')
parser.add_argument('--calibrate', type=str, help='Read sensor data for calibration from dicom file (path to file).')
args = parser.parse_args()

## CT invariables -----------------------------------------------------
# default values
dd_fov_x_rad = b'0.31535198085001725'
dd_fov_y_rad = b'0.24426769480863722'
if args.calibrate is not None:
    if not os.path.isfile(args.calibrate):
        print("ERROR: could not find calibration file: ", args.calibrate)
        exit(1)
    # Read dicom data from existing ct image
    import dicomreader as dr
    print("Reading file: ", args.calibrate)
    dd = dr.read_dicom(args.calibrate)
    dd_fov_x_rad = str(dd.fov_x_rad).encode("UTF-8")
    dd_fov_y_rad = str(dd.fov_y_rad).encode("UTF-8")
    print(dd)


## Renderer setup -----------------------------------------------------
if args.train is not None:
    # Load C++ rendering libs
    stdc      = c.cdll.LoadLibrary("libc.so.6")
    stdcpp    = c.cdll.LoadLibrary("libc++.so.1")
    renderlib = c.cdll.LoadLibrary("./src/match/librender.so")
    # Create and init renderer instance
    create_renderer_wrapper = renderlib.create_renderer
    create_renderer_wrapper.restype = c.c_void_p
    renderer = c.c_void_p(create_renderer_wrapper())
    arg_buffers = [c.create_string_buffer(b"./src/match/match"),
                   c.create_string_buffer(args.train.encode("UTF-8")),
                   c.create_string_buffer(b"-device"),
                   c.create_string_buffer(b"gpu"),
                   c.create_string_buffer(b"-fovx"),
                   c.create_string_buffer(dd_fov_x_rad),
                   c.create_string_buffer(b"-fovy"),
                   c.create_string_buffer(dd_fov_y_rad)
                  ]
    arg_ptrs    = (c.c_char_p * len(arg_buffers))(*map(c.addressof, arg_buffers))
    renderlib.init_renderer(renderer, len(arg_buffers), arg_ptrs)
    get_width_wrapper = renderlib.get_width
    get_height_wrapper = renderlib.get_height
    get_bpp_wrapper = renderlib.get_bpp
    get_width_wrapper.restype = c.c_int
    get_height_wrapper.restype = c.c_int
    get_bpp_wrapper.restype = c.c_int
    renderer_width = c.c_int(get_width_wrapper(renderer))
    renderer_height = c.c_int(get_height_wrapper(renderer))
    renderer_bpp = c.c_int(get_bpp_wrapper(renderer))

def spherical_to_cartesian(spherical):
    r     = spherical[0]
    theta = spherical[1]
    phi   = spherical[2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return [x, y, z]

# returns:
# r > 0
# theta [0, pi]
# phi [0, 2pi]
def cartesian_to_spherical(cartesian):
    x = cartesian[0]
    y = cartesian[1]
    z = cartesian[2]
    r = np.sqrt(x*x + y*y + z*z)
    if r < 1e-5:
        return [0, 0, 0]
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    if phi < 0: # return [0, 2pi] instead of [-pi, pi]
        phi = phi + 2 * np.pi
    return [r, theta, phi]

def angle_to_sincos(angle):
    return [np.sin(angle), np.cos(angle)]

def sincos_to_angle(sincos):
    if sincos[0] >= 0:
        return np.arccos(sincos[1])
    return 2 * np.pi - np.arccos(sincos[1])

def map_camera(camera, eye_dist_max, center_dist_max):
    # transform to spherical coords
    eye    = cartesian_to_spherical(camera[0:3])
    center = cartesian_to_spherical(camera[3:6])
    up     = cartesian_to_spherical(camera[6:9])
    # scale to [0, 1]
    # Note: mapping phi to [sin(phi), cos(phi)] to avoid half open interval.
    eye_mapped    = np.empty(4)
    center_mapped = np.empty(4)
    up_mapped     = np.empty(3) # up is normalized: don't need r
    eye_mapped[0] = eye[0] / eye_dist_max
    eye_mapped[1] = eye[1] / np.pi
    eye_mapped[2:4] = np.divide(np.add(angle_to_sincos(eye[2]), 1), 2)
    center_mapped[0] = center[0] / center_dist_max
    center_mapped[1] = center[1] / np.pi
    center_mapped[2:4] = np.divide(np.add(angle_to_sincos(center[2]), 1), 2)
    up_mapped[0] = up[1] / np.pi
    up_mapped[1:3] = np.divide(np.add(angle_to_sincos(up[2]), 1), 2)
    return np.concatenate((eye_mapped, center_mapped, up_mapped))

def restore_camera(camera, eye_dist_max, center_dist_max):
    eye_mapped    = camera[0:4]
    center_mapped = camera[4:8]
    up_mapped     = camera[8:11]
    # unscale
    eye_mapped[0] *= eye_dist_max
    eye_mapped[1] *= np.pi
    eye_mapped[2:4] = np.subtract(np.multiply(eye_mapped[2:4], 2), 1)
    center_mapped[0] *= center_dist_max
    center_mapped[1] *= np.pi
    center_mapped[2:4] = np.subtract(np.multiply(center_mapped[2:4], 2), 1)
    up_mapped[0] *= np.pi
    up_mapped[1:3] = np.subtract(np.multiply(up_mapped[1:3], 2), 1)
    eye    = [   eye_mapped[0],    eye_mapped[1], sincos_to_angle(   eye_mapped[2:4])]
    center = [center_mapped[0], center_mapped[1], sincos_to_angle(center_mapped[2:4])]
    up     = [               1,     up_mapped[0], sincos_to_angle(    up_mapped[1:3])]
    # spherical to cartesian
    eye_c    = spherical_to_cartesian(eye)
    center_c = spherical_to_cartesian(center)
    up_c     = spherical_to_cartesian(up)
    return np.concatenate((eye_c, center_c, up_c))

def get_frame(camera, integration_coefficient=0.0000034, random_vignette=False, random_integration_coefficient=False):
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
    if random_integration_coefficient == True:
        random_integration_coefficient *= np.random.uniform(0.6, 1.15);
    int_coeff = (c.c_float)(integration_coefficient)
    renderlib.single_shot(renderer, image_buff_ptr, int_coeff, eye_x, eye_y, eye_z, center_x, center_y, center_z, up_x, up_y, up_z)
    if random_vignette == True:
        x_min = np.random.randint(renderer_width.value * 0.2, renderer_width.value * 0.45)
        x_max = np.random.randint(renderer_width.value * 0.55, renderer_width.value * 0.8)
        y_min = np.random.randint(renderer_height.value * 0.2, renderer_height.value * 0.45)
        y_max = np.random.randint(renderer_height.value * 0.55, renderer_height.value * 0.8)
        vignetted = np.full(shape=(renderer_height.value, renderer_width.value, renderer_bpp.value), dtype=np.uint8, fill_value=255)
        vignetted[y_min:y_max, x_min:x_max] = image_buff[y_min:y_max, x_min:x_max]
        image_buff = vignetted
    return image_buff[:, :, 0:3]

def init_gl():
    renderlib.init_gl()

## Generators ----------------------------------------------------------
eye_dist_max = 2000
center_dist_max = 100

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

    def sample_on_unit_sphere(self):
        'Sample point on 3-dim unit sphere'
        vec = np.random.standard_normal(size=3)
        vec /= np.linalg.norm(vec, axis=0)
        return vec

    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.uint8)
        y = np.empty((self.batch_size, 11), dtype=np.float32)

        #TODO seems to be running in new thread: need to re-init gl(ew).
        init_gl()

        # Generate data
        for i in range(self.batch_size):
            #TODO get volume dims.
            volume_radius = 500
            volume_center = [0, 0, 0]
            eye_search_radius_inner = 1400
            eye_search_radius_outer = 1800
            center_search_radius = 100

            eye    = volume_center + self.sample_on_unit_sphere() * np.random.uniform(eye_search_radius_inner, eye_search_radius_outer)
            center = volume_center + self.sample_on_unit_sphere() * np.random.uniform(0, center_search_radius)
            up     = self.sample_on_unit_sphere()

            #TODO chances that up and dir are colinear?...
            cam_dir = np.subtract(center, eye)
            orthogonal = np.cross(cam_dir, up)
            up = np.cross(cam_dir, orthogonal)
            up /= np.linalg.norm(up)

            true_camera   = np.concatenate((eye, center, up))
            mapped_camera = map_camera(true_camera, eye_dist_max, center_dist_max)

            y[i] = mapped_camera
            X[i,] = get_frame(true_camera, random_vignette=True, random_integration_coefficient=True)
            #import cv2 as cv
            #cv.namedWindow("Display Image", cv.WINDOW_AUTOSIZE);
            #cv.imshow("Display Image", X[i,]);
            #cv.waitKey(0);
        
        X = tf.keras.applications.inception_resnet_v2.preprocess_input(X, data_format='channels_last')

        return X, y


## Model ---------------------------------------------------------------
# Parameters
dim_y = renderer_width.value
dim_x = renderer_height.value
params = {'dim': (dim_x, dim_y),
          'batch_size': 16,
          'n_channels': 3}  #n_channels must be 3 for resnet

# Setup model
model = tf.keras.models.Sequential()
loss_weights = None
if args.load is not None:
    if not os.path.isfile(args.load):
        print("ERROR: could not find pre-trained model file: ", args.load)
        exit(1)
    print('loading trained model...')
    model = tf.keras.models.load_model('trained_model.keras')
else:
    print('No pre-trained model specified. Creating new model...')
    model.add(tf.keras.layers.Resizing(height=224, width=224, interpolation='bilinear', crop_to_aspect_ratio=True))
    resnet = tf.keras.applications.InceptionResNetV2(include_top=False,
                       #input_shape=(dim_x, dim_y, 3),
                       input_shape=(224, 224, 3),
                       pooling='None',
                       weights='imagenet')
    for layer in resnet.layers:
        layer.trainable=False
    model.add(resnet)
    model.add(tf.keras.layers.AveragePooling2D((3,3)))
    model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(11, activation='sigmoid'))
    #model.add(tf.keras.layers.Dense(11, activation='tanh'))
    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', loss_weights=loss_weights, metrics=['mean_squared_error'])
    model.build(input_shape=(None, dim_x, dim_y, 3))

# Generators
training_generator   = DataGenerator(batches_per_epoch=32, **params)
validation_generator = DataGenerator(batches_per_epoch=4, **params)

def plot_step(fit_history, epochs, learning_rate, index):
    fig, ax = plotter_lib.subplots()
    ax.plot(range(epochs), fit_history.history['mean_squared_error'], label='Training MSE')
    ax.plot(range(epochs), fit_history.history['val_mean_squared_error'], label='Validation MSE')
    ax.set(xlabel='Epochs', ylabel='MSE', title='Step ' + str(index) + ', learning rate = ' + str(learning_rate))
    ax.legend()
    fig.savefig('mse' + str(index) + '.png')
    #plotter_lib.show()

def run_step(model, training_generator, validation_generator, step, epochs, learning_rate):
    #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error', loss_weights=loss_weights, metrics=['mean_squared_error'])
    #optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', loss_weights=loss_weights, metrics=['mean_squared_error'])
    model.summary()
    print("Learning rate: ", model.optimizer.learning_rate.numpy())
    fit_history = model.fit(x=training_generator, validation_data=validation_generator, epochs=epochs, shuffle=False, use_multiprocessing=False, workers=1)
    plot_step(fit_history=fit_history, epochs=epochs, learning_rate=learning_rate, index=step)
    

## Training ------------------------------------------------------------
if args.train is not None:
    step = 1
    epochs = 60
    learning_rate=1e-3
    print("freeze resnet layers...")
    model.get_layer(name='inception_resnet_v2').trainable=False
    run_step(model, training_generator, validation_generator, step, epochs, learning_rate)
    
    step = 2
    epochs = 150
    learning_rate=1e-3
    print("train resnet layers...")
    model.get_layer(name='inception_resnet_v2').trainable=True
    run_step(model, training_generator, validation_generator, step, epochs, learning_rate)
    
    step = 3
    epochs = 150
    learning_rate=1e-4
    run_step(model, training_generator, validation_generator, step, epochs, learning_rate)
    
    step = 4
    epochs = 150
    learning_rate=1e-5
    run_step(model, training_generator, validation_generator, step, epochs, learning_rate)


## Prediction ----------------------------------------------------------
if args.predict is not None:
    #TODO loop over prediction files
    print("predict...")
    eye = [1670, 0, 0]
    center = [0, 0, 0]
    up = [0, 1, 0]
    test_cam = np.concatenate((eye, center, up))
    mapped_test_cam = map_camera(test_cam, eye_dist_max, center_dist_max)
    test_image = get_frame(test_cam, random_vignette=True, random_integration_coefficient=False)
    #import cv2 as cv
    #cv.namedWindow("Display Image", cv.WINDOW_AUTOSIZE);
    #cv.imshow("Display Image", test_image);
    #cv.waitKey(0);
    preprocessed_test_image = tf.keras.applications.inception_resnet_v2.preprocess_input(np.expand_dims(test_image, axis=0), data_format='channels_last')
    test_prediction=model(preprocessed_test_image, training=False)
    print("Test: eye=", eye, " center=", center, " up=", up)
    predicted_cam = restore_camera(test_prediction[0, 0:11].numpy(), eye_dist_max, center_dist_max)
    print("Pred: eye=", predicted_cam[0:3], " center=", predicted_cam[3:6], " up=", predicted_cam[6:9])

    # export as json
    #TODO
    #if args.export_predictions is not None:
        #TODO

## Save Model ----------------------------------------------------------
if args.store is not None:
    print("Saving model as: ", args.store)
    model.save(args.store)

## Cleanup -------------------------------------------------------------
if args.train is not None:
    renderlib.destroy_renderer(renderer)
