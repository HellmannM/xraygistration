Localize objects in two (non-collinear) X-Ray images in 3D CT data with markerless image registration.

## Project structure

### Viewer application [Anari-DRR-Viewer](https://github.com/HellmannM/anari-drr-viewer)
- Core of the project:
    * Render DRRs from CT data with anari-visionary
    * Pixel picker for X-ray images
    * Find transform between X-rays and rendered DRRs with an Image-Transform-Estimator
    * Localize selected pixels in CT data
- Loads at runtime:
    * Renderer (anari-visionaray with DRR extension)
    * Image-Transform-Estimator (selectable at runtime)
    * 3D CT data as Nifti file
    * X-ray images as PNG files
    * Initial values / estimations for perspectives of X-rays
    * Sensor / device information for both CT and X-ray imaging machines
- CT and X-ray file types chosen intentionally enforce anonymity. No patient data is required.
  Relevant sensor information may be extracted from DICOM beforehand.

### [Image-Transform-Estimator](https://github.com/HellmannM/image-transform-estimator)
- Implementations of multiple algorithms to calculate the perspective transformation between images
- Feature based
    * ORB detector + ORB descriptor
    * SURF detector + SIFT descriptor
- Intensity based
    * Enhanced Correlation Coefficient

### [DRR Renderer](https://github.com/HellmannM/anari-visionaray)
- Modified version of [anari-visionaray](https://github.com/szellmann/anari-visionaray) with DRR extension
- Implements ANARI and offers CPU and GPU (Nvidia + AMD) devices
- DRR extension is essentially a ray marcher that uses [Linear Accumulation Coefficients](https://www.researchgate.net/publication/51657849_A_method_to_produce_and_validate_a_digitally_reconstructed_radiograph-based_computer_simulation_for_optimisation_of_chest_radiographs_acquired_with_a_computed_radiography_imaging_system)

### DICOM-Reader
- Read dicom metadata from X-Ray and extract relevant information such as:
    * Emitter dimensions
    * Detector dimensions and resolution
    * Distance emitter to detector

### DNN-Match
- Calibrate renderer with data from DICOM-Reader
- Train DNN to return camera params by feeding it DRRs
- Predict camera params from X-Rays with trained DNN
- Current DNN structure:
    * Layers:
        + Resize image (to fixed resolution)
        + Inception-Resnet-V2 (exclude top, no pooling, init with imagenet weights)
        + AveragePooling2D (3, 3)
        + Flatten
        + Dense(11, activation='sigmoid')
    * Optimizer: Adam
    * Loss: MSE
    * Learning rate: 1e-3 gradually decreasing to 1e-5
    * Exclude Inception-Resnet-V2 from training in first epochs
    * Batch size: 16
    * Batches per epoch: 32 training, 4 validation


## Workflow overview

### As soon as CT volumetric data is available:
- Extract sensor data from DICOM metadata for X-Ray and CT imaging machines to calibrate DRR generation.
- Train AI (DNNMatch) on DRRs from CT volumetric data to perform rough image registration
### Once X-Rays are available:
- Get rough estimation of camera params for both X-Rays from AI (DNNMatch)
- Use SolvePnP to fine-tune camera params
- Mark objects (pixels) to be tracked
- Get object (pixel) positions in CT volumetric data


## Limitations / Challenges

- Objects to be tracked are visible in X-Rays but not in CT data. These regions must thus be masked off or at least not influence used algorithms.
- DRRs will never match true X-Rays exactly.
- X-Rays might be heavily occluded (surgical instruments)
- X-Ray images are greyscale and contain little texture which negatively impacts feature extraction algorithms


## Getting started
```Shell
git clone https://github.com/HellmannM/xraygistration.git --recursive
./setup.sh
mkdir build && cd build
cmake ..
make -j24
```


## Command examples
- Launch viewer:
    * `LD_LIBRARY_PATH=./anari-visionaray ./anariDRRViewer ct.nii -l visionaray_cuda -j xrays.json -m ./feature-matcher/libfeature-matcher-surf.so --lacfile ./LacLuts.json --lut 1`
- DNN-Match
    * Train
        `TF_GPU_ALLOCATOR=cuda_malloc_async python3 ./aimatch.py --train ct.nii --save trained_model`
    * Predict
        `TF_GPU_ALLOCATOR=cuda_malloc_async python3 ./aimatch.py --load trained_model.keras --predict image1.png,image2.png --crop 0,0,150,150/0,0,300,150 --export_predictions predictions.json`


## Licensing
See [license file](license.md).


## Dependencies
- [ANARI SDK](https://github.com/KhronosGroup/ANARI-SDK)
- [Visionaray](https://github.com/szellmann/visionaray)
- [ITK](https://github.com/InsightSoftwareConsortium/ITK)
- [CmdLine](https://github.com/abolz/CmdLine)
- [nlohmann json](https://github.com/nlohmann/json)
- [OpenCV](https://github.com/opencv/opencv)
- [Keras](https://github.com/keras-team/keras)

