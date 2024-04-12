## Goal of this project

Localize objects from two (non-collinear) X-Ray images in CT-volumetric data with markerless image registration.


## Basic workflow

### When CT volumetric data is ready:
- Extract sensor data from DICOM metadata in (previous) X-Rays to calibrate DRR generation.
- Train AI (DNNMatch) on DRRs from CT volumetric data to perform rough image registration
### When X-Rays are ready:
- Get rough estimation of camera params for both X-Rays from AI (DNNMatch)
- Use SolvePnP to fine-tune camera params
- Mark objects (pixels) to be tracked
- Get object (pixel) positions in CT volumetric data


## Limitations / Constraints

- Objects to be tracked are visible in X-Rays but not in CT data. These regions must thus be masked off or at least not influence used algorithms.
- DRRs will never match true X-Rays exactly.
- X-Rays might be heavily occluded (surgical instruments)
- X-Ray images are greyscale and contain little texture which negatively impacts feature extraction algorithms


## Project structure

### DICOM-Reader
- Read dicom metadata from X-Ray and extract relevant information such as:
    * Emitter dimensions
    * Detector dimensions and resolution
    * Distance emitter to detector

### Renderer (located within Match)
- Generate DRRs

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


### Match
- Extract features from images
    * ORB detector + ORB descriptor
    * TODO: SURF detector + SIFT descriptor
- Match features from from pair of images
- Reconstruct perspective change between pair of images
    * 3D-2D: SolvePnP
        + Augment 2D Features in DRRs to 3D space (estimate depth value)
        + Calculate perspective change with SolvePnP on 2D-Features from X-Ray and (augmented) 3D-Features from DRR
    * 2D-2D: Homography:
        + Calculate homography between DRR and X-Ray (both 2D images).
        + Estimate perspective change from homography matrix (rotation ok but translation lacks scale)
- Iterative search
    * Generate DRR with current estimation
    * Update current estimation with reconstructed perspective change

## Licensing
See [license file](license.md).

## Dependencies
- [Visionaray](https://github.com/szellmann/visionaray)
- [Deskvox](https://github.com/deskvox/deskvox)
- [CmdLine](https://github.com/abolz/CmdLine)
- [nlohmann json](https://github.com/nlohmann/json)
- [OpenCV](https://github.com/opencv/opencv)
- [Keras](https://github.com/keras-team/keras)

## Current work in progress
See [todo file](todo.md)

