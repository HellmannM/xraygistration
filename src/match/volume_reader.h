#pragma once

#include <string>

using voxel_value_type = int16_t;

class volume_reader
{
    public:
        volume_reader(std::string filename);
        ~volume_reader();

        float size(size_t i) const;

        float voxel_size(size_t i) const;

        size_t dimensions(size_t i) const;

        float origin(size_t i) const;

        voxel_value_type value(int x, int y, int z);

    private:
        struct impl;
        impl* p_impl;
};

