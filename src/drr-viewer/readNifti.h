// Copyright 2024 Matthias Hellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdio.h>
// ITK
#include <itkImageFileReader.h>
// ours
#include "FieldTypes.h"

struct NiftiReader
{
  bool open(const char *fileName);
  const StructuredField &getField(int index = 0);

  using voxel_value_type = int16_t; //TODO
  using img_t = itk::Image<voxel_value_type, 3>;
  using reader_t = itk::ImageFileReader<img_t>;

  typename reader_t::Pointer  reader;
  typename img_t::Pointer     img;
  StructuredField             field;
};
