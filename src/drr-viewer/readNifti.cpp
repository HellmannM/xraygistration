// Copyright 2024 Matthias Hellmann
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <iostream>
#include <vector>
// ITK
#include <itkImageFileReader.h>
#include "itkImageRegionIterator.h"
// ours
#include "attenuation.h"
#include "FieldTypes.h"
#include "readNifti.h"

bool NiftiReader::open(const char *fileName)
{
  auto len = std::strlen(fileName);
  if ((len < 3) || (std::strncmp(fileName + len - 3, "nii", 3) != 0)) {
    return false;
  }

  std::cout << "Reading Nifi file... ";
  reader = reader_t::New();
  reader->SetFileName(fileName);
  reader->Update();
  img = reader->GetOutput();

  field.dimX = reader->GetImageIO()->GetDimensions(0);
  field.dimY = reader->GetImageIO()->GetDimensions(1);
  field.dimZ = reader->GetImageIO()->GetDimensions(2);
  field.bytesPerCell = sizeof(float);

  std::cout << "[" << field.dimX << ", " << field.dimY << ", " << field.dimZ << "]\n";

  return true;
}

const StructuredField& NiftiReader::getField(int index)
{
  if (field.empty()) {
    // transform from ct density to linear attenuation coefficient
    std::cout << "\tTransform density values to linear attenuation coefficients\n";
    using voxel_value_type = int16_t; //TODO
    std::vector<voxel_value_type> buffer;
    itk::ImageRegionConstIterator<img_t> inputIterator(img, img->GetLargestPossibleRegion());
    while (!inputIterator.IsAtEnd())
    {
        buffer.push_back(inputIterator.Get());
        ++inputIterator;
    }
    std::vector<float> attenuation_volume((size_t)field.dimX * field.dimY * field.dimZ);
    for (size_t i=0; i<buffer.size(); ++i) {
      attenuation_volume[i] = attenuation_lookup(buffer[i], tube_potential::TB13000EV);
    }
    size_t size = attenuation_volume.size() * sizeof(attenuation_volume[0]);
    field.dataF32.resize(size);
    memcpy((char *)field.dataF32.data(), attenuation_volume.data(), size);

    field.dataRange = {0.f, 3.f}; //TODO
  }
  return field;
}
