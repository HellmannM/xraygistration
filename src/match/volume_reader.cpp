#include <itkImageFileReader.h>

#include <string>
#include <vector>

#include "volume_reader.h"

using pixel_type = int16_t;
using img_t = itk::Image<pixel_type, 3>;
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

std::string volume_reader::pixel_type_as_str() const
{
    return p_impl->img_reader->GetImageIO()->GetPixelTypeAsString(
            p_impl->img_reader->GetImageIO()->GetPixelType());
}

pixel_type volume_reader::value(int x, int y, int z)
{
    typename img_t::IndexType index{{x, y, z}};
    return p_impl->img->GetPixel(index);
}

//template <typename pixel_type>
//class volume_reader
//{
//    public:
//        using pixel_t = pixel_type;
//        using img_t = itk::Image<pixel_t, 3>;
//        using img_reader_t = itk::ImageFileReader<img_t>;
//        using const_it_t = itk::ImageRegionConstIterator<img_t>;
//
//        nifti_reader(std::string filename)
//        {
//            img_reader = img_reader_t::New();
//            img_reader->SetFileName(filename);
//            img_reader->Update();
//
//            img = img_reader->GetOutput();
//            img_it = const_it_t(img, img->GetRequestedRegion());
//        }
//
//        const_it_t const_image_iterator() { return img_it; }
//
//        template <int i>
//        float size() const
//        {
//            return dimensions<i>() * voxel_size<i>();
//        }
//
//        template <int i>
//        float voxel_size() const
//        {
//            return img_reader->GetImageIO()->GetSpacing(i);
//        }
//
//        template <int i>
//        size_t dimensions() const
//        {
//            return img_reader->GetImageIO()->GetDimensions(i);
//        }
//
//        template <int i>
//        float origin() const
//        {
//            return img_reader->GetImageIO()->GetOrigin(i);
//        }
//
//        std::string pixel_type_as_str() const
//        {
//            return img_reader->GetImageIO()->GetPixelTypeAsString(
//                    img_reader->GetImageIO()->GetPixelType());
//        }
//
//        pixel_type value(int x, int y, int z)
//        {
//            typename img_t::IndexType index{x, y, z};
//            return img->GetPixel(index);
//        }
//
//    private:
//        std::string filename;
//        typename img_reader_t::Pointer img_reader;
//        typename img_t::Pointer img;
//        const_it_t img_it;
//};

