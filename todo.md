# CMake
- [ ] Fix sources (headers)

# anari-drr-viewer
- [ ] Export JSON
- [ ] Pixel selection
- [ ] Match code
- [ ] Find light-weight nifti and dicom loader
- [ ] Fix loading/displaying PNGs with other pixel format than RGBA8
- [ ] Use multiple samples per pixel or force center of pixel (compare with visRtx single-shot commit)
- [ ] Make visionaray and nlohmann json mandatory
- [ ] Fix rendering bug in drr-viewer: last row is on top

# anari-visionaray
- [ ] Add DICOM image loader
- [ ] Fix bug when replacing volume (switch LAC LUT at runtime)

# Match
### Matcher:
- [ ] Check License for SURF
- [ ] Fix cuda::ORB constructor in template specialization
- [ ] Move renderer to own header
### Renderer:
- [ ] Simulate emitter surface area (instead of pinhole cam)
- [ ] add routine to tune brightness & contrast to match input image
- [ ] respect voxel dims
- [ ] add scatter mask + noise
### Search:
- [x] current approach:
    * generate image with current cam
    * match with input image
    * update camera
    * loop
- [ ] new approach:
    * generate multiple images with randomized cam within certain range of current cam
    * match all with input image
    * exclude outliers and average remaining results
    * update camera
    * loop

