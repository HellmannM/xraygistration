# dependencies
- [ ] move itk to anari-drr-viewer
- [ ] move opencv to ite
- [ ] nlohmann, cmdline?

# anari-drr-viewer
- [ ] Find light-weight nifti and dicom loader
- [ ] Use multiple samples per pixel or force center of pixel (compare with visRtx single-shot commit)
- [ ] auto-adjust brightness?
- [ ] multiple 'screenshot' routines. consolidate and use either visionaray or stb-image but not both.

# anari-visionaray
- [ ] Add DICOM image loader
- [~] fix memory leak when re-comitting renderer (calls commit on field)
- [~] scatter: is simply subtracting enough?

# feature-matcher
- [ ] match many images at once, filter closest images, update cam with average of remaining results?

# image-based matcher
- [x] Mutual Information:
        - OpenCV findTransformECC with MOTION_HOMOGRAPHY
- [ ] Dense optical flow
        - OpenCV Fernebäck (calcOpticalFlowFerneback)
        - RAFT Optical Flow (robust but need to train DNN)

