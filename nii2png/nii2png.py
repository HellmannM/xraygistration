import argparse as argparse
import imageio as iio
import numpy as np
import os as os
import nibabel as nib

parser = argparse.ArgumentParser(description='Convert nii to png.')
parser.add_argument('-i', type=str, help='nii input file.')
parser.add_argument('-o', type=str, help='output file.')
args = parser.parse_args()

inputfile = str(args.i)
outputfile = str(args.o)

if not os.path.isfile(inputfile):
    print("Input file does not exist: ", inputfile)
    exit(1)

if args.o is None:
    outputfile = inputfile[:-3] + "png"

if os.path.isfile(outputfile):
    print("Output file already exists: ", outputfile)
    exit(1)

print("Converting " + inputfile + " to " + outputfile)

img = nib.load(inputfile)
data = img.get_fdata()

print("Shape: ", data.shape)
print("Dims: ", data.ndim)
#_min = np.amin(data)
#_max = np.amax(data)
#print("min: ", _min)
#print("max: ", _max)
data = data.astype(np.int32) # int16 should be enough but doesn't work for some reason...
data = data * 16

if data.ndim == 2:
    iio.imwrite(outputfile, data)

#if data.ndim == 3:
#    for i in range(0, img.shape[3]):
#        outputfile_base = outputfile[:-4]
#        #get slice
#        imgslice = img[::i]
#        #check if slice file exists
#        outputfile = outputfile_base + str(i) + ".png"
#        if os.path.isfile(outputfile):
#            print("Output file already exists: ", outputfile)
#            exit(1)
#        # write slice
#        iio.imwrite(outputfile, imgsilce)

