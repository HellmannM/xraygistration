# CMake
- [ ] Fix sources (headers)

# anari-drr-viewer
- [ ] Export JSON
- [ ] Pixel selection & calc code
- [ ] Find light-weight nifti and dicom loader
- [ ] Fix loading/displaying PNGs with other pixel format than RGBA8
- [ ] Use multiple samples per pixel or force center of pixel (compare with visRtx single-shot commit)
- [ ] Load FOV from JSON and adjust viewport (also match image resolution?)
- [ ] auto-adjust brightness?
- [ ] update load camera after matching or add new button
- [ ] add FOV slider

# anari-visionaray
- [ ] Add DICOM image loader
- [ ] Fix bug when replacing volume (switch LAC LUT at runtime)
- [ ] add scatter mask + noise
- [ ] simulate emitter surface instead of pinhole cam? multiple samples with jittered origin?
- [ ] pinhole camera: are rectangular pixels really enough? check xray dicoms...

# feature-matcher
- [ ] match many images at once, filter closest images, update cam with average of remaining results?
- [ ] fix focal length calc.
      Allow only rectangular viewport as a workaround? -> use fovy and height only

