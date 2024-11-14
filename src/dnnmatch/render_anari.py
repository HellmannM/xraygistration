# Copyright 2021-2024 The Khronos Group
# SPDX-License-Identifier: Apache-2.0

from anari import *
import argparse as argparse
from lac_transform import *
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os as os

# Load Nifti
parser = argparse.ArgumentParser(description='Render nifti file.')
parser.add_argument('-i', type=str, help='nii input file.')
args = parser.parse_args()
inputfile = str(args.i)
if not os.path.isfile(inputfile):
    print("Input file does not exist: ", inputfile)
    exit(1)
img = nib.load(inputfile)
data = img.get_fdata()
print("Shape: ", data.shape)
data = data.astype(np.float32)

print("Transform to LAC...")
data_lac = transform_to_lac_multiproc(data).astype(np.float32)

print("Render...")
# ANARI
prefixes = {
    lib.ANARI_SEVERITY_FATAL_ERROR : "FATAL",
    lib.ANARI_SEVERITY_ERROR : "ERROR",
    lib.ANARI_SEVERITY_WARNING : "WARNING",
    lib.ANARI_SEVERITY_PERFORMANCE_WARNING : "PERFORMANCE",
    lib.ANARI_SEVERITY_INFO : "INFO",
    lib.ANARI_SEVERITY_DEBUG : "DEBUG"
}

def anari_status(device, source, sourceType, severity, code, message):
    print('[%s]: '%prefixes[severity]+message)

status_handle = ffi.new_handle(anari_status) #something needs to keep this handle alive

debug = anariLoadLibrary('debug', status_handle)
library = anariLoadLibrary('visionaray', status_handle)

nested = anariNewDevice(library, 'default')
anariCommitParameters(nested, nested)

device = anariNewDevice(debug, 'debug')
anariSetParameter(device, device, 'wrappedDevice', ANARI_DEVICE, nested)
anariSetParameter(device, device, 'traceMode', ANARI_STRING, 'code')
anariCommitParameters(device, device)

# Camera ------------
width = 500
height = 384
cam_pos = [255.5, 609.667, -770.566]
cam_up = [0.0, 1.0, 0.0]
cam_dir = [0.0, -0.34202, 0.939693]
cam_fov_x_rad = 0.31535198085001725
cam_fov_y_rad = 0.24426769480863722
camera = anariNewCamera(device, 'perspective')
anariSetParameter(device, camera, 'aspect', ANARI_FLOAT32, width/height)
anariSetParameter(device, camera, 'position', ANARI_FLOAT32_VEC3, cam_pos)
anariSetParameter(device, camera, 'direction', ANARI_FLOAT32_VEC3, cam_dir)
anariSetParameter(device, camera, 'up', ANARI_FLOAT32_VEC3, cam_up)
anariSetParameter(device, camera, 'fovy', ANARI_FLOAT32, cam_fov_y_rad)
#TODO can't set fovx -> modify aspect?
anariCommitParameters(device, camera)

world = anariNewWorld(device)

spatialField = anariNewSpatialField(device, 'structuredRegular')
array = anariNewArray3D(device, ffi.from_buffer(data_lac), ANARI_FLOAT32, data_lac.shape[0], data_lac.shape[1], data_lac.shape[2])
anariSetParameter(device, spatialField, 'data', ANARI_ARRAY3D, array)
anariSetParameter(device, spatialField, 'filter', ANARI_STRING, 'linear');
anariCommitParameters(device, spatialField)

volume = anariNewVolume(device, 'transferFunction1D')
anariSetParameter(device, volume, 'value', ANARI_SPATIAL_FIELD, spatialField)
colors = np.array([
  0.0, 0.0, 1.0,
  0.0, 1.0, 0.0,
  1.0, 0.0, 0.0
], dtype = np.float32)
opacities = np.array([
  0.0, 1.0
], dtype = np.float32)
array = anariNewArray1D(device, ffi.from_buffer(colors), ANARI_FLOAT32_VEC3, 3)
anariSetParameter(device, volume, 'color', ANARI_ARRAY1D, array)
array = anariNewArray1D(device, ffi.from_buffer(opacities), ANARI_FLOAT32, 2)
anariSetParameter(device, volume, 'opacity', ANARI_ARRAY1D, array)
anariCommitParameters(device, volume)

volumes = ffi.new('ANARIVolume[]', [volume])
array = anariNewArray1D(device, volumes, ANARI_VOLUME, 1)
anariSetParameter(device, world, 'volume', ANARI_ARRAY1D, array)
anariCommitParameters(device, world)

renderer = anariNewRenderer(device, 'drr')
bg_color = [0.8, 0.4, 0.4, 1.0]
anariSetParameter(device, renderer, 'background', ANARI_FLOAT32_VEC4, bg_color)
anariCommitParameters(device, renderer)

frame = anariNewFrame(device)
anariSetParameter(device, frame, 'size', ANARI_UINT32_VEC2, [width, height])
anariSetParameter(device, frame, 'channel.color', ANARI_DATA_TYPE, ANARI_UFIXED8_RGBA_SRGB)
anariSetParameter(device, frame, 'renderer', ANARI_RENDERER, renderer)
anariSetParameter(device, frame, 'camera', ANARI_CAMERA, camera)
anariSetParameter(device, frame, 'world', ANARI_WORLD, world)
anariCommitParameters(device, frame)

anariRenderFrame(device, frame)
anariFrameReady(device, frame, ANARI_WAIT)
void_pixels, frame_width, frame_height, frame_type = anariMapFrame(device, frame, 'channel.color')

unpacked_pixels = ffi.buffer(void_pixels, frame_width*frame_height*4)
pixels = np.array(unpacked_pixels).reshape([height, width, 4])
plt.imshow(pixels)
plt.show()
anariUnmapFrame(device, frame, 'channel.color')

