#pragma once

#include <cstdint>
#include <string>

namespace visionaray {

struct dimensions
{
    uint64_t x, y, z;
};

class fileio
{
public:
    fileio() = default;

    fileio(std::string file)
    :   filename(file)
      , byte_dims()
      , voxel_dims()
      , num_channels()
      , bytes_per_channel()
      , bytes_per_voxel()
      , data(nullptr)
    {}

    //TODO impl cpy & move ctors

    ~fileio()
    {
        try
        {
            delete data;
        } catch(...) {}
        data = nullptr;
    }


    void load(const std::string& file)
    {
        filename = file;
        load();
    }

    void save(const std::string& file)
    {
        filename = file;
        save();
    }

    std::string get_filename() const noexcept { return filename; }
    dimensions get_byte_dimensions() const noexcept { return byte_dims; }
    dimensions get_voxel_dimensions() const noexcept { return voxel_dims; }
    uint64_t byte_dim_x() const noexcept { return byte_dims.x; }
    uint64_t byte_dim_y() const noexcept { return byte_dims.y; }
    uint64_t byte_dim_z() const noexcept { return byte_dims.z; }
    uint64_t voxel_dim_x() const noexcept { return voxel_dims.x; }
    uint64_t voxel_dim_y() const noexcept { return voxel_dims.y; }
    uint64_t voxel_dim_z() const noexcept { return voxel_dims.z; }
    uint8_t* get_data() noexcept { return data; }

    uint8_t* get_voxel(size_t x, size_t y, size_t z)
    {
        uint8_t* raw = data;
        raw += z * (voxel_dims.x * voxel_dims.y * bytes_per_voxel);
        return &raw[(voxel_dims.x * y + x) * bytes_per_voxel];
    }

    const uint8_t* get_voxel(size_t x, size_t y, size_t z) const
    {
        const uint8_t* raw = data;
        raw += z * (voxel_dims.x * voxel_dims.y * bytes_per_voxel);
        return &raw[(voxel_dims.x * y + x) * bytes_per_voxel];
    }

protected:
    std::string     filename;
    dimensions      byte_dims;
    dimensions      voxel_dims;
    uint32_t        num_channels;
    uint32_t        bytes_per_channel;
    uint32_t        bytes_per_voxel;
    uint8_t*        data;

private:
    virtual void load() = 0;
    virtual void save() = 0;
};

} // namespace visionaray

// forward include
#include "nifti.h"
