# Match
### Matcher:
- [ ] Use SURF Detector + SIFT Descriptor instead of ORB.
- [ ] License for SURF/SIFT
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
- [ ] add canny edge filter before feeding to net?
- [ ] repeatable test: add fixed (instead of random) crop to test image
- [ ] research normalized flow / flow matching