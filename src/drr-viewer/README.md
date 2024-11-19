# ANARI Digitally reconstructed radiograph (DRR) viewer.

## This project builds upon the ANARI Mini-Viewer for Volumetric Data:
https://github.com/vtvamr/anari-volume-viewer

Dependencies:

- ANARI-SDK: https://github.com/KhronosGroup/ANARI-SDK/ (version 0.8.x, with
`INSTALL_VIEWER_LIBRARY=ON`)
- ITK (optional, support for nifti volume files)

## Usage:

```
anariDRRViewer [{--help|-h}]
   [{--verbose|-v}] [{--debug|-g}]
   [{--library|-l} <ANARI library>]
   [{--trace|-t} <directory>]
   [{--dims|-d} <dimx dimy dimz>]
   [{--type|-t} [{uint8|uint16|float32}]
   <volume file>
```


## License

Apache 2 (if not noted otherwise)
