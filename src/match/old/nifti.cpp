#include <cassert>
#include <cfloat>
#include <climits>
#include <iostream>
#include <stdexcept>

#include <nifti1_io.h>

#include <visionaray/math/vector.h>

#include "nifti.h"

namespace visionaray {

void nifti_fileio::load()
{
    bool verbose = true;

    // read file header
    nifti_image* header = nifti_image_read(filename.c_str(), 0);
    if (!header)
    {
        throw std::runtime_error("nifti fileio");
    }

    // extract dimensions
    voxel_dims.x = header->nx;
    voxel_dims.y = header->ny;
    voxel_dims.z = header->nz;
    byte_dims.x = header->dx;
    byte_dims.y = header->dy;
    byte_dims.z = header->dz;

    // extract voxel format
    vec2f mapping;
    switch (header->datatype)
    {
    case NIFTI_TYPE_RGB24:
        if (verbose) std::cout << "Datatype: NIFTI_TYPE_RGB24\n";
        num_channels = 3;
        bytes_per_channel = header->nbyper / 3;
        break;
    case NIFTI_TYPE_RGBA32:
        if (verbose) std::cout << "Datatype: NIFTI_TYPE_RGB32\n";
        num_channels = 4;
        bytes_per_channel = header->nbyper / 4;
        break;
    case NIFTI_TYPE_INT8:
        if (verbose) std::cout << "Datatype: NIFTI_TYPE_INT8\n";
        assert(header->nbyper == 1);
        num_channels = 1;
        bytes_per_channel = header->nbyper;
        mapping = {CHAR_MIN, CHAR_MAX};
        break;
    case NIFTI_TYPE_UINT8:
        if (verbose) std::cout << "Datatype: NIFTI_TYPE_UINT8\n";
        assert(header->nbyper == 1);
        num_channels = 1;
        bytes_per_channel = header->nbyper;
        mapping = {0, UCHAR_MAX};
        break;
    case NIFTI_TYPE_INT16:
        if (verbose) std::cout << "Datatype: NIFTI_TYPE_INT16\n";
        assert(header->nbyper == 2);
        num_channels = 1;
        bytes_per_channel = header->nbyper;
        mapping = {SHRT_MIN, SHRT_MAX};
        break;
    case NIFTI_TYPE_UINT16:
        if (verbose) std::cout << "Datatype: NIFTI_TYPE_UINT16\n";
        assert(header->nbyper == 2);
        num_channels = 1;
        bytes_per_channel = header->nbyper;
        mapping = {0, USHRT_MAX};
        break;
    case NIFTI_TYPE_INT32:
        if (verbose) std::cout << "Datatype: NIFTI_TYPE_INT32\n";
        assert(header->nbyper == 4);
        num_channels = 1;
        bytes_per_channel = header->nbyper;
        mapping = {(float)INT_MIN, (float)INT_MAX};
        break;
    case NIFTI_TYPE_UINT32:
        if (verbose) std::cout << "Datatype: NIFTI_TYPE_UINT32\n";
        assert(header->nbyper == 4);
        num_channels = 1;
        bytes_per_channel = header->nbyper;
        mapping = {0, (float)UINT_MAX};
        break;
    case NIFTI_TYPE_FLOAT32:
        if (verbose) std::cout << "Datatype: NIFTI_TYPE_FLOAT32\n";
        assert(header->nbyper == 4);
        num_channels = 1;
        bytes_per_channel = header->nbyper;
        mapping = {-FLT_MAX, FLT_MAX};
        break;
    default:
        if (verbose) std::cout << "Datatype: UNKNOWN\n";
        num_channels = 1;
        bytes_per_channel = header->nbyper;
        break;
    }


    // read image data ------------------------------------
    nifti_image* data_section = nifti_image_read(filename.c_str(), 1);
    if (!data_section)
    {
        throw std::runtime_error("nifti fileio");
    }

    bytes_per_voxel = bytes_per_channel * num_channels;
    auto frame_bytes = byte_dims.x * byte_dims.y * byte_dims.z * bytes_per_voxel;
    data = new uint8_t[frame_bytes];
    memcpy(data, static_cast<uint8_t*>(data_section->data), frame_bytes);

    float slope = header->scl_slope;
    float inter = header->scl_inter;

    if (verbose)
    {
        std::cout << "Intercept: " << inter << ", slope: " << slope << '\n';
    }


    // adapt data formats
    if (header->datatype == NIFTI_TYPE_INT16)
    {
        mapping = {SHRT_MIN * slope + inter, SHRT_MAX * slope + inter};

        // Remap data
        for (size_t z = 0; z < voxel_dims.z; ++z)
        {
            for (size_t y = 0; y < voxel_dims.y; ++y)
            {
                for (size_t x = 0; x < voxel_dims.x; ++x)
                {
                    uint8_t* bytes = get_voxel(x, y, z);
                    int32_t voxel = (int)*reinterpret_cast<int16_t*>(bytes);
                    voxel -= SHRT_MIN;
                    *reinterpret_cast<uint16_t*>(bytes) = voxel;
                }
            }
        }
    }
    else if (header->datatype == NIFTI_TYPE_INT32)
    {
        mapping = {INT_MIN * slope + inter, INT_MAX * slope + inter};

        // Remap data to float
        for (size_t z = 0; z < voxel_dims.z; ++z)
        {
            for (size_t y = 0; y < voxel_dims.y; ++y)
            {
                for (size_t x = 0; x < voxel_dims.x; ++x)
                {
                    uint8_t* bytes = get_voxel(x, y, z);
                    int32_t i = *reinterpret_cast<int32_t*>(bytes);
                    float f = static_cast<float>(i);
                    *reinterpret_cast<float*>(bytes) = f;
                }
            }
        }
    }
    else if (header->datatype == NIFTI_TYPE_UINT32)
    {
        mapping = {inter, UINT_MAX * slope + inter};

        // Remap data to float
        for (size_t z = 0; z < voxel_dims.z; ++z)
        {
            for (size_t y = 0; y < voxel_dims.y; ++y)
            {
                for (size_t x = 0; x < voxel_dims.x; ++x)
                {
                    uint8_t* bytes = get_voxel(x, y, z);
                    unsigned u = *reinterpret_cast<unsigned*>(bytes);
                    float f = static_cast<float>(u);
                    *reinterpret_cast<float*>(bytes) = f;
                }
            }
        }
    }
    else
    {
        mapping *= slope;
        mapping += inter;
    }

    // read value ranges for tf
//    for (int c = 0; c < num_channels; ++c)
//    {
//        vd->findMinMax(c, vd->range(c)[0], vd->range(c)[1]);
//        vd->tf[c].setDefaultColors(vd->getChan() == 1 ? 0 : 4 + c, vd->range(c)[0], vd->range(c)[1]);
//        vd->tf[c].setDefaultAlpha(0, vd->range(c)[0], vd->range(c)[1]);
//    }
}

void nifti_fileio::save()
{
    int dimensions[] = { 3, (int)voxel_dims.x, (int)voxel_dims.y, (int)voxel_dims.z, 1, 0, 0, 0 };

    int datatype = 0;
    if (bytes_per_channel == 1)
        datatype = NIFTI_TYPE_UINT8;
    else if (bytes_per_channel == 2)
        datatype = NIFTI_TYPE_UINT16;
    else if (bytes_per_channel == 4)
        datatype = NIFTI_TYPE_FLOAT32;

    nifti_image* img = nifti_make_new_nim(dimensions, datatype, 1);

    nifti_set_filenames(img, filename.c_str(), 0, 0);

    img->dx = byte_dims.x;
    img->dy = byte_dims.y;
    img->dz = byte_dims.z;

    img->data = get_data();

    nifti_image_write(img);
}

} // namespace visionaray
