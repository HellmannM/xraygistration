# dicomreader.py

import argparse
import numpy as np
import os.path
import png
import pydicom
import sys

class dicom_data:
    def __init__(self,
            sensor_width_px, sensor_height_px, sensor_width_mm, sensor_height_mm,
            image_width_px, image_height_px, image_width_mm, image_height_mm,
            fov_x_rad, fov_y_rad, fov_x_deg, fov_y_deg):
        self.sensor_width_px = sensor_width_px
        self.sensor_height_px = sensor_height_px
        self.sensor_width_mm = sensor_width_mm
        self.sensor_height_mm = sensor_height_mm
        self.image_width_px = image_width_px
        self.image_height_px = image_height_px
        self.image_width_mm = image_width_mm
        self.image_height_mm = image_height_mm
        self.fov_x_rad = fov_x_rad
        self.fov_y_rad = fov_y_rad
        self.fov_x_deg = fov_x_deg
        self.fov_y_deg = fov_y_deg

    def __repr__(self):
        return "dicom_data: class containing dicom data"

    def __str__(self):
        return "DICOM Data:\n" \
            + "\nSensor size:\n" \
            + "width: " + str(self.sensor_width_px)  + "px\n" \
            + "height:" + str(self.sensor_height_px) + "px\n" \
            + "width: " + str(self.sensor_width_mm)  + "mm\n" \
            + "height:" + str(self.sensor_height_mm) + "mm\n" \
            + "\nImage size:\n" \
            + "width: " + str(self.image_width_px)   + "px\n" \
            + "height:" + str(self.image_height_px)  + "px\n" \
            + "width: " + str(self.image_width_mm)   + "mm\n" \
            + "height:" + str(self.image_height_mm)  + "mm\n" \
            + "\nFOV:\n" \
            + "FOV X:" + str(self.fov_x_rad)         + "rad\n" \
            + "FOV Y:" + str(self.fov_y_rad)         + "rad\n" \
            + "FOV X:" + str(self.fov_x_deg)         + "°\n"   \
            + "FOV Y:" + str(self.fov_y_deg)         + "°\n"

def read_entry(ds, group, element, default_value=0):
    try:
        return ds[group, element].value
    except:
        print(f'Error: could not find [{group:#06X}, {element:#06X}]')
        return default_value

def read_header(ds):
    dist_source_detector        = read_entry(ds, 0x0018, 0x1110, 1)
#    dist_source_patient         = read_entry(ds, 0x0018, 0x1111)
#    dist_source_entrance        = read_entry(ds, 0x0040, 0x0306)
    imager_pixel_spacing        = read_entry(ds, 0x0018, 0x1164, 1)
#    positioner_primary_angle    = read_entry(ds, 0x0018, 0x1510)
#    positioner_secondary_angle  = read_entry(ds, 0x0018, 0x1511)
    rows                        = read_entry(ds, 0x0028, 0x0010, 1)
    columns                     = read_entry(ds, 0x0028, 0x0011, 1)
    left_edge                   = read_entry(ds, 0x0018, 0x1602, 0)
    right_edge                  = read_entry(ds, 0x0018, 0x1604, 1)
    upper_edge                  = read_entry(ds, 0x0018, 0x1606, 0)
    lower_edge                  = read_entry(ds, 0x0018, 0x1608, 1)

    sensor_width_mm = columns * imager_pixel_spacing[1] #TODO at 0 or 1?
    sensor_height_mm = rows * imager_pixel_spacing[0] #TODO at 0 or 1?
    image_width = right_edge - left_edge + 1
    image_height = lower_edge - upper_edge + 1
    image_width_mm = image_width * imager_pixel_spacing[1] #TODO at 0 or 1?
    image_height_mm = image_height * imager_pixel_spacing[0] #TODO at 0 or 1?
    fov_x = 2 * np.arctan(image_width_mm  / 2 / dist_source_detector)
    fov_y = 2 * np.arctan(image_height_mm / 2 / dist_source_detector)
    fov_x_deg = fov_x / ( 2 * np.pi ) * 360
    fov_y_deg = fov_y / ( 2 * np.pi ) * 360

    dd = dicom_data(
        sensor_width_px  = columns,
        sensor_height_px = rows,
        sensor_width_mm  = sensor_width_mm,
        sensor_height_mm = sensor_height_mm,
        image_width_px   = image_width,
        image_height_px  = image_height,
        image_width_mm   = image_width_mm,
        image_height_mm  = image_height_mm,
        fov_x_rad        = fov_x,
        fov_y_rad        = fov_y,
        fov_x_deg        = fov_x_deg,
        fov_y_deg        = fov_y_deg
        )
    
    #print(f"{dist_source_detector.value=}")
    #print(f"{dist_source_patient.value=}")
    #print(f"{dist_source_entrance.value=}")
    #print(f"{imager_pixel_spacing.value=}")
    #print(f"{positioner_primary_angle.value=}")
    #print(f"{positioner_secondary_angle.value=}")
    #print(f"{rows.value=}")
    #print(f"{columns.value=}")
    #print(f"{left_edge.value=}")
    #print(f"{right_edge.value=}")
    #print(f"{upper_edge.value=}")
    #print(f"{lower_edge.value=}")

    return dd

def export_as_png(ds, export_path, max_value=4095):
    bits_stored = read_entry(ds, 0x0028, 0x0101, 12)
    max_value = pow(2, bits_stored)

    image_2d = ds.pixel_array.astype(float)
    image_2d = (np.maximum(image_2d,0) / max_value) * 255.0
    image_2d = np.uint8(image_2d)

    writer = png.Writer(image_2d.shape[1], image_2d.shape[0], greyscale=True)
    writer.write(export_path, image_2d)
    print(f'PNG exported to: "{export_path.name}"\n')
    return

def main() -> int:
    parser = argparse.ArgumentParser(description='Extract info from DICOM header and optionally export as PNG.')
    parser.add_argument('dicom', type=str, help='Path to DICOM file.')
    parser.add_argument('--png', type=str, help='Export as PNG (path to file).')
    parser.add_argument('--force', action=argparse.BooleanOptionalAction, default=False, help='Force export (overwrite files if necessary).')
    args = parser.parse_args()

    if not os.path.isfile(args.dicom):
        raise Exception('File not found: "%s"' % args.dicom)
    dicom_file = open(args.dicom, 'rb')
    ds = pydicom.dcmread(dicom_file)
    dicom_file.close()

    # read header
    dd = read_header(ds)
    print(dd)

    # export as png
    if args.png is not None:
        if os.path.isfile(args.png):
            if not args.force:
                raise Exception('File exists and would be overwritten: "%s"' % args.png)
            os.remove(args.png)
        png_file = open(args.png, 'wb')
        export_as_png(ds, png_file)
        png_file.close()

    return 0

if __name__ == "__main__":
    main()


#(0018, 1134) Table Motion                        CS: 'STATIC'
#(0018, 1152) Exposure                            IS: None
#(0018, 1155) Radiation Setting                   CS: 'GR'
#(0018, 1600) Shutter Shape                       CS: 'RECTANGULAR'
#(0028, 0004) Photometric Interpretation          CS: 'MONOCHROME2'
#(0028, 1050) Window Center                       DS: '509.0'
#(0028, 1051) Window Width                        DS: '683.0'
#(2001, 1074) [Window Center Sub]                 DS: '589.0'
#(2001, 1075) [Window Width Sub]                  DS: '512.0'
#(2001, 1077) [GL TrafoType]                      CS: 'LINEARVOI'
#(2001, 102c) [Harmonization Factor]              FL: 0.5
#(2001, 102f) [Harmonization Gain]                FL: 1.2000000476837158
#(2001, 104f) [Harmonization Offset]              FD: 0.2199999988079071
#(2001, 109f) [Pixel Processing Kernel Size]      US: [43, 43]
#(0018, 9469) Table Horizontal Rotation Angle     FL: 0.0
#(0018, 9470) Table Head Tilt Angle               FL: 0.0
#(0018, 9471) Table Cradle Tilt Angle             FL: 0.0
#(300a, 0128) Table Top Vertical Position         DS: '983.0'
#(300a, 0129) Table Top Longitudinal Position     DS: '323.0'
#(300a, 012a) Table Top Lateral Position          DS: '13.0'
