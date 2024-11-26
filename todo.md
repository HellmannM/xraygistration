# CMake
- [ ] Fix sources (headers)

# anari-drr-viewer
- [ ] Export JSON
- [ ] Pass depth estimation through
- [ ] Pixel selection
- [ ] Match code
- [ ] Possibility to set camera
- [ ] Load (and display?) xray files

# Match
### Matcher:
- [ ] Check License for SURF
- [ ] Fix cuda::ORB constructor in template specialization
- [ ] Move renderer to own header
- [x] Better depth estimation / back projection
      maybe copy from "A CNN Regression Approach for Real-Time 2D 3D Registration"
- [x] use LAC LUTs for reprojection/depth estimation
### Renderer:
- [ ] Simulate emitter surface area (instead of pinhole cam)
- [ ] add routine to tune brightness & contrast to match input image
- [ ] respect voxel dims
- [x] calc traveled distance
- [x] variable tube potential
- [ ] add scatter mask + noise
- [x] use different nifti loader? maybe ITK
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


# DNNMATCH
- [x] research normalized flow / flow matching
