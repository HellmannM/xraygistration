# main.py

import ctypes as c
import numpy as np
import sys

# Load C++ libs
stdc      = c.cdll.LoadLibrary("libc.so.6")
stdcpp    = c.cdll.LoadLibrary("libc++.so.1")
renderlib = c.cdll.LoadLibrary("./src/match/librender.so")

create_renderer_wrapper = renderlib.create_renderer
create_renderer_wrapper.restype = c.c_void_p
renderer = c.c_void_p(create_renderer_wrapper())
print('p renderer=', renderer)

arg_buffers = [c.create_string_buffer(b"./src/match/match"),
               c.create_string_buffer(b"../testfiles/Dummy_Paul_nifti/2__head_10_stx_head.nii"),
               c.create_string_buffer(b"-device"),
               c.create_string_buffer(b"gpu")
              ]
arg_ptrs    = (c.c_char_p*4)(*map(c.addressof, arg_buffers))
renderlib.init_renderer(renderer, 4, arg_ptrs)
print('p renderer=', renderer)

get_width_wrapper = renderlib.get_width
get_height_wrapper = renderlib.get_height
get_bpp_wrapper = renderlib.get_bpp
get_width_wrapper.restype = c.c_int
get_height_wrapper.restype = c.c_int
get_bpp_wrapper.restype = c.c_int
width = c.c_int(renderlib.get_width())
height = c.c_int(renderlib.get_height())
bpp = c.c_int(renderlib.get_bpp())
print('width=', width)
print('height=', height)
print('bpp=', bpp)
sys.exit()

image_buff = c.create_string_buffer(width * height * bpp)
print('p renderer=', renderer)
print('before: ', image_buff[0], ', ', image_buff[1], ', ', image_buff[2], ', ', image_buff[3])
eye_x = (c.c_float)(1000)
eye_y = (c.c_float)(0)
eye_z = (c.c_float)(0)
center_x = (c.c_float)(0)
center_y = (c.c_float)(0)
center_z = (c.c_float)(0)
up_x = (c.c_float)(0)
up_y = (c.c_float)(1)
up_z = (c.c_float)(0)
renderlib.single_shot(
        renderer,
        image_buff,
        eye_x,
        eye_y,
        eye_z,
        center_x,
        center_y,
        center_z,
        up_x,
        up_y,
        up_z)
print('after: ', image_buff[0], ', ', image_buff[1], ', ', image_buff[2], ', ', image_buff[3])
