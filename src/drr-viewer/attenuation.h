#pragma once

#include <cstdint>
#include <vector>

enum tube_potential
{
    TB13000EV = 0,
    TB13500EV = 1,
    TB14000EV = 2
};

struct lut_entry { ssize_t density; float lac; };

std::vector<std::vector<lut_entry>> LAC_LUT = {
    { // 13000 eV
        {0,    0.0},
        {21,   0.000289},
        {282,  0.0747},
        {443,  0.11205},
        {909,  0.13708},
        {911,  0.09568},   
        {970,  0.18216},
        {1016, 0.153},
        {1027, 0.230945},
        {1029, 0.159355},
        {1113, 0.17135},
        {1114, 0.29268},
        {1143, 0.1888},
        {1228, 0.3744},
        {1301, 0.5544},
        {1329, 1.06485},
        {1571, 0.90852},
        {2034, 1.6224},
        {2516, 2.484}
    },
    { // 13500 eV
        {0,    0.0},
        {21,   0.00026},
        {282,  0.0672},
        {443,  0.1008},
        {909,  0.1242},
        {911,  0.087492},
        {970,  0.16434},
        {1016, 0.13872},
        {1027, 0.207955},
        {1029, 0.14413},
        {1113, 0.1514},
        {1114, 0.26244},
        {1143, 0.1711},
        {1228, 0.33813},
        {1301, 0.4984},
        {1329, 0.958365},
        {1571, 0.8174},
        {2034, 1.45704},
        {2516, 2.244}
    },
    { // 14000 eV
        {0,    0.0},
        {21,   0.000235},
        {282,  0.0609},
        {443,  0.09135},
        {909,  0.11316},
        {911,  0.080224},
        {970,  0.14949},
        {1016, 0.12648},
        {1027, 0.1881},
        {1029, 0.130935},
        {1113, 0.1403},
        {1114, 0.2376},
        {1143, 0.15576},
        {1228, 0.30537},
        {1301, 0.45024},
        {1329, 0.864475},
        {1571, 0.73834},
        {2034, 1.31508},
        {2516, 2.02}
    }
};


float attenuation_lookup(ssize_t density, tube_potential tb/*, size_t& count_below_0, size_t& count_above_2516*/) {
    if (density < 0)
    {
//        ++count_below_0;
        density = 0;
    }
    else if (density > 2516)
    {
//        ++count_above_2516;
        density = 2516;
    }
    const auto pos = find_if(LAC_LUT[tb].begin(), LAC_LUT[tb].end(),
            [density](lut_entry elem){return elem.density > density;}
            );
    const auto previous = pos - 1;
    return previous->lac + (density - previous->density) / (pos->density - previous->density) * (pos->lac - previous->lac);
}

