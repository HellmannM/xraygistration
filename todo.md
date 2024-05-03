# CMake
- [ ] Fix sources (headers)

# Match
### Matcher:
- [ ] Check License for SURF
- [ ] Fix cuda::ORB constructor in template specialization
- [ ] Move renderer to own header
### Renderer:
- [ ] Simulate emitter surface area (instead of pinhole cam)
- [ ] add routine to tune brightness & contrast to match input image
### Search:
- [ ] current approach:
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
- [ ] research normalized flow / flow matching
