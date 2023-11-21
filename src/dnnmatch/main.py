# main.py

import ctypes
#from ctypes import cdll

# Load C++ libs
stdc      = ctypes.cdll.LoadLibrary("libc.so.6")
stdcpp    = ctypes.cdll.LoadLibrary("libc++.so.1")
renderlib = ctypes.cdll.LoadLibrary("./src/match/librender.so")

renderer = renderlib.create_renderer()
renderer_wrapper = renderlib.create_renderer
renderer_wrapper.restype = ctypes.c_void_p
renderer = ctypes.c_void_p(renderer_wrapper())

arg_buffers = [ctypes.create_string_buffer(256) for i in range(2)]
arg_buffers = ["./src/match/match\0", "../testfiles/Dummy_Paul_nifti/2__head_10_stx_head.nii\n"]
arg_ptrs    = (ctypes.c_char_p*2)(*map(ctypes.addressof, arg_buffers))
renderlib.init_renderer(renderer, 2, arg_ptrs)
