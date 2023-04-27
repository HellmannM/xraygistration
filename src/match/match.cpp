// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <common/config.h>

#include <cstring> // memcpy
#include <exception>
#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

#include <GL/glew.h>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp> // cv::Mat
#include <opencv2/features2d.hpp> // cv::ORB

#if VSNRAY_COMMON_HAVE_CUDA
// CUDA includes
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#endif

// CmdLine includes
#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

// Visionaray includes
#undef MATH_NAMESPACE
#include <visionaray/detail/platform.h>
#include <visionaray/math/math.h>
#include <visionaray/texture/texture.h>
#include <visionaray/pinhole_camera.h>
#include <visionaray/scheduler.h>

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/viewer_glut.h>

// Deskvox includes
#undef MATH_NAMESPACE
#include <virvo/vvfileio.h>
#include <virvo/vvpixelformat.h>
#include <virvo/vvtextureutil.h>

#include "host_device_rt.h"
#include "render.h"

using namespace visionaray;
using viewer_type   = viewer_glut;


//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer : viewer_type
{
    renderer()
        : viewer_type(1024, 1024, "Visionaray Volume Rendering Example")
        , bbox({ -1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f, 1.0f })
        , rt(
            host_device_rt::CPU,
            true /* double buffering */,
            false /* direct rendering */,
            host_device_rt::RGB
            )
        , host_sched(8)
#if VSNRAY_COMMON_HAVE_CUDA
        , device_sched(8, 8)
#endif
        , volume_ref({std::array<unsigned int, 3>({2, 2, 2})})
        , filename()
        , texture_format(virvo::PF_R16I)
        , delta(0.01f)
        , bgcolor({1.f, 1.f, 1.f})
        , orb(cv::ORB::create(
                /*int nfeatures     */ 5000,
                /*float scaleFactor */ 1.2f,
                /*int nlevels       */ 8,
                /*int edgeThreshold */ 31,
                /*int firstLevel    */ 0,
                /*int WTA_K         */ 2,
                /*int scoreType     */ cv::ORB::HARRIS_SCORE,
                /*int patchSize     */ 31,
                /*int fastThreshold */ 15
          ))
        , matcher(cv::BFMatcher::create(cv::NORM_HAMMING, true))
        , matcher_initialized(false)
    {
        // Add cmdline options
        add_cmdline_option( support::cl::makeOption<std::string&>(
                    support::cl::Parser<>(),
                    "filename",
                    support::cl::Desc("Input file in nii format"),
                    support::cl::Positional,
                    support::cl::Required,
                    support::cl::init(filename)
                    ) );

#if VSNRAY_COMMON_HAVE_CUDA
        add_cmdline_option( support::cl::makeOption<host_device_rt::mode_type&>({
                { "cpu", host_device_rt::CPU, "Rendering on the CPU" },
                { "gpu", host_device_rt::GPU, "Rendering on the GPU" },
            },
            "device",
            support::cl::Desc("Rendering device"),
            support::cl::ArgRequired,
            support::cl::init(rt.mode())
            ) );
#endif

    }

    aabb                                                bbox;
    pinhole_camera                                      cam;
    host_device_rt                                      rt;
    tiled_sched<ray_type_cpu>                           host_sched;
    volume_t                                            volume;
    volume_ref_t                                        volume_ref;
#if VSNRAY_COMMON_HAVE_CUDA
    cuda_sched<ray_type_gpu>                            device_sched;
    cuda_volume_t                                       device_volume;
    cuda_volume_ref_t                                   device_volume_ref;
#endif

    std::string                                         filename;
    vvVolDesc*                                          vd;
    // Internal storage format for textures
    virvo::PixelFormat                                  texture_format;
    float                                               delta;
    vec3                                                bgcolor;
    vec2f                                               value_range;

    cv::Ptr<cv::ORB>                                    orb;
    cv::Ptr<cv::BFMatcher>                              matcher;
    bool                                                matcher_initialized;
    std::vector<vector<4, unorm<8>>>                    reference_image_std;
    cv::Mat                                             reference_image;
    cv::Mat                                             reference_descriptors;
    std::vector<cv::KeyPoint>                           reference_keypoints;

    void load_volume();
    void update_reference_image();
    void match();
    std::vector<vector<4, unorm<8>>> get_current_image();
protected:

    void on_display();
    void on_resize(int w, int h);
    void on_key_press(visionaray::key_event const& event);

};


//-------------------------------------------------------------------------------------------------
// Display function, implements the volume rendering algorithm
//

void renderer::on_display()
{
    if (rt.mode() == host_device_rt::CPU)
    {
        render_cpp(
                volume_ref,
                bbox,
                value_range,
                rt,
                host_sched,
                cam,
                delta
            );
    }
#if VSNRAY_COMMON_HAVE_CUDA
    else if (rt.mode() == host_device_rt::GPU)
    {
        render_cu(
                device_volume_ref,
                bbox,
                value_range,
                rt,
                device_sched,
                cam,
                delta
            );
    }
#endif

    // display the rendered image
    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    rt.swap_buffers();
    rt.display_color_buffer();
}


//-------------------------------------------------------------------------------------------------
// resize event
//
void renderer::on_resize(int w, int h)
{
    cam.set_viewport(0, 0, w, h);
    float aspect = w / static_cast<float>(h);
    cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    rt.resize(w, h);

    viewer_type::on_resize(w, h);
}

//-------------------------------------------------------------------------------------------------
// key event
//
void renderer::on_key_press(key_event const& event)
{
    switch (event.key())
    {
    case 'c':
        if (rt.color_space() == host_device_rt::RGB)
        {
            rt.color_space() = host_device_rt::SRGB;
        }
        else
        {
            rt.color_space() = host_device_rt::RGB;
        }
        break;

   case 'm':
#if VSNRAY_COMMON_HAVE_CUDA
        if (rt.mode() == host_device_rt::CPU)
        {
            std::cout << "Switching to GPU.\n";
            rt.mode() = host_device_rt::GPU;
        }
        else
        {
            std::cout << "Switching to CPU.\n";
            rt.mode() = host_device_rt::CPU;
        }
#endif
        break;

    case 'r':
        std::cout << "Updating reference image.\n";
        update_reference_image();
        break;

    case 't':
        std::cout << "Matching...\n";
        match();
        break;

    default:
        break;
    }

    viewer_type::on_key_press(event);
}

//-------------------------------------------------------------------------------------------------
// load volume if file
//
void renderer::load_volume()
{
    std::cout << "Loading volume file: " << filename << std::endl;
    vd = new vvVolDesc(filename.c_str());
    vvFileIO fio;
    if (fio.loadVolumeData(vd) != vvFileIO::OK)
    {
        std::cerr << "Error loading volume" << std::endl;
        delete vd;
        vd = NULL;
        return;
    }
    else vd->printInfoLine();
    virvo::TextureUtil tu(vd);
    assert(vd->getChan() == 1); // only support single channel data
    virvo::TextureUtil::Pointer tex_data = nullptr;
    virvo::TextureUtil::Channels channelbits = 1ULL;
    tex_data = tu.getTexture(virvo::vec3i(0),
            virvo::vec3i(vd->vox),
            texture_format,
            channelbits,
            0 /*frame*/ );
    // update vol
    volume = volume_t(vd->vox[0], vd->vox[1], vd->vox[2]);
    //volume.reset(reinterpret_cast<volume_ref_t::value_type const*>(tex_data));
    volume.reset(reinterpret_cast<volume_value_t const*>(vd->getRaw(0)));
//    volume.reset(reinterpret_cast<volume_value_t const*>(tex_data));
    volume_ref = volume_ref_t(volume);
    volume_ref.set_filter_mode(Nearest);
    volume_ref.set_address_mode(Clamp);
#if VSNRAY_COMMON_HAVE_CUDA
    std::cout << "Copying volume and transfer function to gpu.\n";
    device_volume = cuda_volume_t(volume_ref);
    device_volume.set_filter_mode(Nearest);
    device_volume.set_address_mode(Clamp);
    device_volume_ref = cuda_volume_ref_t(device_volume);
#endif

    bbox = aabb(vec3(vd->getBoundingBox().min.data()), vec3(vd->getBoundingBox().max.data()));

    // determine ray integration step size (aka delta)
    int axis = 0;
    if (vd->getSize()[1] / vd->vox[1] < vd->getSize()[axis] / vd->vox[axis])
    {
        axis = 1;
    }
    if (vd->getSize()[2] / vd->vox[2] < vd->getSize()[axis] / vd->vox[axis])
    {
        axis = 2;
    }
    //TODO expose quality variable
    int quality = 1.0f;
    delta = (vd->getSize()[axis] / vd->vox[axis]) / quality;
    std::cout << "Using delta=" << delta << "\n";

    value_range = vec2f(vd->range(0).x, vd->range(0).y);
    std::cout << "Dataset value range: min=" << value_range.x << " max=" << value_range.y << "\n";
    value_range = vec2f(-900, vd->range(0).y);
    std::cout << "Clamping to:         min=" << value_range.x << " max=" << value_range.y << "\n";
}

std::vector<vector<4, unorm<8>>> renderer::get_current_image()
{
    std::vector<vector<4, unorm<8>>> rgba(rt.width() * rt.height());
    
    if (rt.mode() == host_device_rt::CPU)
    {
        memcpy(rgba.data(), rt.color(host_device_rt::buffer::Front), rt.width() * rt.height() * sizeof(vector<4, unorm<8>>));
    }
#if VSNRAY_COMMON_HAVE_CUDA
    else if (rt.mode() == host_device_rt::GPU)
    {
        cudaMemcpy(
            rgba.data(),
            rt.color(host_device_rt::buffer::Front),
            rt.width() * rt.height() * sizeof(vector<4, unorm<8>>),
            cudaMemcpyDeviceToHost
            );
    }
#endif

//    swizzle(
//        rgb.data(),
//        PF_RGB8,
//        rt.color(),
//        PF_RGBA32F,
//        //PF_RGB8,
//        rt.width() * rt.height(),
//        TruncateAlpha
//        );
//
//    if (rt.color_space() == host_device_rt::SRGB)
//    {
//        for (int y = 0; y < rt.height(); ++y)
//        {
//            for (int x = 0; x < rt.width(); ++x)
//            {
//                auto& color = rgb[y * rt.width() + x];
//                color.x = powf(color.x, 1 / 2.2f);
//                color.y = powf(color.y, 1 / 2.2f);
//                color.z = powf(color.z, 1 / 2.2f);
//            }
//        }
//    }

    // Flip so that origin is (top|left)
    std::vector<vector<4, unorm<8>>> flipped(rt.width() * rt.height());

    for (int y = 0; y < rt.height(); ++y)
    {
        for (int x = 0; x < rt.width(); ++x)
        {
            int yy = rt.height() - y - 1;
            flipped[yy * rt.width() + x] = rgba[y * rt.width() + x];
        }
    }

    return flipped;
}

void renderer::update_reference_image()
{
    //auto reference_image_std = get_current_image();
    //auto reference_image = cv::Mat(rt.height(), rt.width(), CV_8UC4, reinterpret_cast<void*>(reference_image_std.data()));
    reference_image_std = get_current_image();
    reference_image = cv::Mat(rt.height(), rt.width(), CV_8UC4, reinterpret_cast<void*>(reference_image_std.data()));

    //std::vector<cv::KeyPoint> reference_keypoints;
    //cv::Mat reference_descriptors;
    reference_keypoints.clear();
    orb->detectAndCompute(reference_image, cv::noArray(), reference_keypoints, reference_descriptors);
    std::cout << "Found " << reference_descriptors.size() << " descriptors.\n";

    matcher->clear();
    matcher->add(reference_descriptors);

    matcher_initialized = true;
}

void renderer::match()
{
    if (!matcher_initialized) return;

    auto current_image_std = get_current_image();
    auto current_image = cv::Mat(rt.height(), rt.width(), CV_8UC4, reinterpret_cast<void*>(current_image_std.data()));

    std::vector<cv::KeyPoint> current_keypoints;
    cv::Mat current_descriptors;
    orb->detectAndCompute(current_image, cv::noArray(), current_keypoints, current_descriptors);
    std::cout << "Found " << current_descriptors.size() << " descriptors.\n";
    std::vector<cv::DMatch> matches;
    matcher->match(current_descriptors, matches, cv::noArray());
    std::cout << "Found " << matches.size() << " matches.\n";
    float match_ratio = (float)matches.size() / reference_descriptors.size().height;
    //std::sort(matches.begin(), matches.end(), [](const cv::DMatch& lhs, const cv::DMatch& rhs){ return lhs.distance < rhs.distance;});
    float distance = 0.f;
    for (auto& m : matches)
    {
        distance += m.distance;
        //std::cout << m.distance << "\n";
    }
    std::cout << "Match ratio: " << match_ratio << "\t Average distance: " << distance/matches.size() << "\n";
    //std::cout << "total match distance: " << distance << "\n";
    //cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    //cv::Mat img;
    //cv::drawMatches(current_image, current_keypoints, reference_image, reference_keypoints, matches, img);
    //cv::imshow("Display Image", img);
    //cv::waitKey(0);
}

//-------------------------------------------------------------------------------------------------
// Main function, performs initialization
//

int main(int argc, char** argv)
{
    renderer rend;

    try
    {
        rend.init(argc, argv);
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
    rend.load_volume();

    float aspect = rend.width() / static_cast<float>(rend.height());

    rend.cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    rend.cam.view_all( rend.bbox );

    rend.add_manipulator( std::make_shared<arcball_manipulator>(rend.cam, mouse::Left) );
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Middle) );
    // Additional "Alt + LMB" pan manipulator for setups w/o middle mouse button
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Left, keyboard::Alt) );
    rend.add_manipulator( std::make_shared<zoom_manipulator>(rend.cam, mouse::Right) );

    rend.event_loop();
}
