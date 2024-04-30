#pragma once

#include <iostream>
#include <string>

#include <visionaray/math/vector.h>

struct prediction
{
    std::string filename;
    visionaray::vector<3, float> eye;
    visionaray::vector<3, float> center;
    visionaray::vector<3, float> up;

    prediction(std::string file,
                float eye_x,    float eye_y,    float eye_z,
                float center_x, float center_y, float center_z,
                float up_x,     float up_y,     float up_z)
        : filename(file), eye(eye_x, eye_y, eye_z), center(center_x, center_y, center_z), up(up_x, up_y, up_z) {}
};

std::ostream& operator<<(std::ostream& os, const prediction& p)
{
    return os << p.filename << "\neye: " << p.eye << "\ncenter: " << p.center << "\nup: " << p.up << "\n";
}
