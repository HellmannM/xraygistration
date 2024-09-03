#include <itkImageFileReader.h>
#include "itkImageRegionIterator.h"

#include <string>
#include <vector>

#include "volume_reader.h"

using img_t = itk::Image<voxel_value_type, 3>;
using img_reader_t = itk::ImageFileReader<img_t>;

struct volume_reader::impl
{
        typename img_reader_t::Pointer img_reader;
        typename img_t::Pointer img;
};


volume_reader::volume_reader(std::string filename)
: p_impl(new impl())
{
    p_impl->img_reader = img_reader_t::New();
    p_impl->img_reader->SetFileName(filename);
    p_impl->img_reader->Update();

    p_impl->img = p_impl->img_reader->GetOutput();
}

volume_reader::~volume_reader()
{
    delete p_impl;
}

float volume_reader::size(size_t i) const
{
    return dimensions(i) * voxel_size(i);
}

float volume_reader::voxel_size(size_t i) const
{
    return p_impl->img_reader->GetImageIO()->GetSpacing(i);
}

size_t volume_reader::dimensions(size_t i) const
{
    return p_impl->img_reader->GetImageIO()->GetDimensions(i);
}

float volume_reader::origin(size_t i) const
{
    return p_impl->img_reader->GetImageIO()->GetOrigin(i);
}

voxel_value_type volume_reader::value(int x, int y, int z)
{
    typename img_t::IndexType index{{x, y, z}};
    return p_impl->img->GetPixel(index);
}

void volume_reader::copy(std::vector<voxel_value_type>& buffer)
{
    itk::ImageRegionConstIterator<img_t> inputIterator(p_impl->img, p_impl->img->GetLargestPossibleRegion());
    while (!inputIterator.IsAtEnd())
    {
        buffer.push_back(inputIterator.Get());
        ++inputIterator;
    }
}
