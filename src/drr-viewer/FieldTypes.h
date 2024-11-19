// Copyright 2023 Stefan Zellmann and Jefferson Amstutz
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <vector>

// Structured field type //////////////////////////////////////////////////////
struct StructuredField
{
  std::vector<uint8_t> dataUI8;
  std::vector<uint16_t> dataUI16;
  std::vector<float> dataF32;
  int dimX{0};
  int dimY{0};
  int dimZ{0};
  unsigned bytesPerCell{0};
  struct
  {
    float x, y;
  } dataRange;

  bool empty() const
  {
    if (bytesPerCell == 1 && dataUI8.empty())
      return true;
    if (bytesPerCell == 2 && dataUI16.empty())
      return true;
    if (bytesPerCell == 4 && dataF32.empty())
      return true;
    return false;
  }
};

