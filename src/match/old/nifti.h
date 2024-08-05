#pragma once

#include "fileio.h"

namespace visionaray {

class nifti_fileio : public fileio
{
public:
    void load(std::string& file);
    void save(std::string& file);
private:
    void load() override;
    void save() override;
};

} // namespace visionaray
