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

#include <common/image.h>
#include <common/manip/arcball.h>
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
#include "match_result.h"
#include "render.h"

using namespace visionaray;
using viewer_type   = viewer_glut;


//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer : viewer_type
{
    enum search_mode { eye = 0, center };

    renderer()
        //: viewer_type(1024, 1024, "Visionaray Volume Rendering Example")
        : viewer_type(500, 384, "Visionaray Volume Rendering Example")
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
        , volume_filename()
        , reference_image_filename()
        , texture_format(virvo::PF_R16I)
        , delta(0.01f)
        , integration_coefficient(0.0000034f)
        , bgcolor({1.f, 1.f, 1.f})
        , orb(cv::ORB::create(                                  // default values
                /*int nfeatures     */ 5000,                    // 500
                /*float scaleFactor */ 1.1f,                    // 1.2f
                /*int nlevels       */ 15,                       // 8
                /*int edgeThreshold */ 71,                      // 31
                /*int firstLevel    */ 0,                       // 0
                /*int WTA_K         */ 2,                       // 2
                /*int scoreType     */ cv::ORB::HARRIS_SCORE,   // cv::ORB::HARRIS_SCORE
                /*int patchSize     */ 71,                      // 31
                /*int fastThreshold */ 20                       // 20
          ))
        , matcher(cv::BFMatcher::create(cv::NORM_HAMMING, true))
        , matcher_initialized(false)
    {
        // Add cmdline options
        add_cmdline_option( support::cl::makeOption<std::string&>(
            support::cl::Parser<>(),
            "volume",
            support::cl::Desc("Volume file in nii format"),
            support::cl::Positional,
            support::cl::Required,
            support::cl::init(volume_filename)
            ) );

        add_cmdline_option( support::cl::makeOption<std::string&>(
            support::cl::Parser<>(),
            "ref",
            support::cl::Desc("Reference image file in nii format"),
            //support::cl::Positional,
            //support::cl::Optional,
            support::cl::ArgRequired,
            support::cl::init(reference_image_filename)
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

    std::string                                         volume_filename;
    std::string                                         reference_image_filename;
    vvVolDesc*                                          vd;
    // Internal storage format for textures
    virvo::PixelFormat                                  texture_format;
    float                                               delta;
    float                                               integration_coefficient;
    vec3                                                bgcolor;
    vec2f                                               value_range;

    cv::Ptr<cv::ORB>                                    orb;
    cv::Ptr<cv::BFMatcher>                              matcher;
    bool                                                matcher_initialized;
    std::vector<vector<4, unorm<8>>>                    reference_image_std;
    std::vector<uint8_t>                                reference_image_std2;
    cv::Mat                                             reference_image;
    cv::Mat                                             reference_descriptors;
    std::vector<cv::KeyPoint>                           reference_keypoints;

    void load_volume();
    void load_reference_image();
    void update_reference_image();
    match_result_t match();
    void search();
    void search_impl(const search_mode mode, const int grid_size, const float search_distance);
    void search_impl_up(const float rotation_range);
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
                delta,
                integration_coefficient
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
                delta,
                integration_coefficient
            );
    }
#endif

    // display the rendered image
    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    rt.swap_buffers();
    rt.display_color_buffer();
}

void renderer::search()
{
    constexpr float relax = 1.1f; // modify search_distance slightly to avoid reusing exact same points.
    constexpr int grid_size = 7;
    constexpr int iterations = 5;

    // intertwined
    float eye_search_distance = cam.distance() * 0.2f;
    float center_search_distance = length(bbox.size()) * 0.2f;
    float rotation_range = 90.f;
    for (int iteration = 1; iteration <= iterations; ++iteration)
    {
        std::cout << "Iteration " << iteration << " of " << iterations << ":\n";
        // eye
        search_impl(search_mode::eye, grid_size, eye_search_distance);
        eye_search_distance = eye_search_distance * 3 / grid_size * relax;
        // center
        search_impl(search_mode::center, grid_size, center_search_distance);
        center_search_distance = center_search_distance * 3 / grid_size * relax;
        // up
        search_impl_up(rotation_range);
        rotation_range *= 0.5f;
    }
}

void renderer::search_impl_up(const float rotation_range_deg)
{
    auto best_cam = cam;
    const auto initial_cam = cam;
    // start with current camera
    auto best_result = match();

    constexpr int steps = 40;
    const float angle = rotation_range_deg / steps;
    const vec3 eye    = initial_cam.eye();
    const vec3 center = initial_cam.center();
    const vec3 dir = eye - center;
    const vec3 n = normalize(dir);
    for (int i=0; i<steps; ++i)
    {
        constexpr double pi = M_PI;
        const float cos_a = cos((i * angle - rotation_range_deg / 2) / 360 * 2 * pi);
        const float sin_a = sin((i * angle - rotation_range_deg / 2) / 360 * 2 * pi);
        mat3f rotation_matrix
        {
            n.x * n.x * (1 - cos_a) +       cos_a,    n.x * n.y * (1 - cos_a) - n.z * sin_a,    n.x * n.z * (1 - cos_a) + n.y * sin_a,
            n.y * n.x * (1 - cos_a) + n.z * sin_a,    n.y * n.y * (1 - cos_a) +       cos_a,    n.y * n.z * (1 - cos_a) + n.x * sin_a,
            n.z * n.x * (1 - cos_a) - n.y * sin_a,    n.z * n.y * (1 - cos_a) + n.x * sin_a,    n.z * n.z * (1 - cos_a) +       cos_a
        };
        vec3 up = rotation_matrix * initial_cam.up();

        cam.look_at(eye, center, up);
        on_display();
        const auto current_result = match();
        if (current_result > best_result)
        //if (current_result.match_ratio() > best_result.match_ratio())
        {
            best_result = current_result;
            best_cam = cam;
        }
    }
    cam = best_cam;
    std::cout << " up:     " << best_result.match_ratio() << "\n";
}

void renderer::search_impl(const search_mode mode, const int grid_size, const float search_distance)
{
    const float step_size = 2 * search_distance / (grid_size - 1);

    auto best_cam = cam;
    const auto initial_cam = cam;
    match_result_t best_result;

    float progress = 0;
    for (int x = 0; x < grid_size; ++x)
    {
        for (int y = 0; y < grid_size; ++y)
        {
            for (int z = 0; z < grid_size; ++z)
            {
                vec3 eye    = initial_cam.eye();
                vec3 center = initial_cam.center();
                vec3 up     = initial_cam.up();
                switch (mode)
                {
                case search_mode::eye:
                {
                    eye.x = eye.x - search_distance + (x * step_size);
                    eye.y = eye.y - search_distance + (y * step_size);
                    eye.z = eye.z - search_distance + (z * step_size);
                    break;
                }
                case search_mode::center:
                {
                    center.x = center.x - search_distance + (x * step_size);
                    center.y = center.y - search_distance + (y * step_size);
                    center.z = center.z - search_distance + (z * step_size);
                    break;
                }
                }
                cam.look_at(eye, center, up);
                on_display();
                const auto current_result = match();
                if (current_result > best_result)
                {
                    best_result = current_result;
                    best_cam = cam;
                }
            }
        }
//        // show progress
//        progress += 100.f / grid_size;
//        std::cout << "\r" << progress << " %";
//        std::cout.flush();
    }
    cam = best_cam;
    switch (mode)
    {
    case search_mode::eye:
    {
        std::cout << " eye:    " << best_result.match_ratio() << "\n";
        break;
    }
    case search_mode::center:
    {
        std::cout << " center: " << best_result.match_ratio() << "\n";
        break;
    }
    }
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

    case 's':
        std::cout << "Searching best match...\n";
        search();
        break;

    case 't':
        std::cout << "Matching...\n";
        match();
        break;

    case '+':
        integration_coefficient += 0.0000001f;
        std::cout << "Integration coefficient = " << integration_coefficient << "\n";
        break;

    case '-':
        integration_coefficient -= 0.0000001f;
        std::cout << "Integration coefficient = " << integration_coefficient << "\n";
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
    std::cout << "Loading volume file: " << volume_filename << "\n";
    vd = new vvVolDesc(volume_filename.c_str());
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

#if 0
    swizzle(
        rgb.data(),
        PF_RGB8,
        rt.color(),
        PF_RGBA32F,
        //PF_RGB8,
        rt.width() * rt.height(),
        TruncateAlpha
        );

    if (rt.color_space() == host_device_rt::SRGB)
    {
        for (int y = 0; y < rt.height(); ++y)
        {
            for (int x = 0; x < rt.width(); ++x)
            {
                auto& color = rgb[y * rt.width() + x];
                color.x = powf(color.x, 1 / 2.2f);
                color.y = powf(color.y, 1 / 2.2f);
                color.z = powf(color.z, 1 / 2.2f);
            }
        }
    }
#endif

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

void renderer::load_reference_image()
{
    if (reference_image_filename.empty())
    {
        std::cout << "No reference image file provided.\n";
    }
    std::cout << "Loading reference image file: " << reference_image_filename << "\n";
    visionaray::image img;
    img.load(reference_image_filename);
    std::cout << "width=" << img.width() << " height=" << img.height() << " pf=" << img.format() << "\n";
    on_resize(img.width(), img.height());

    int bpp = 4; //TODO
    reference_image_std2.clear();
    reference_image_std2.resize(img.width() * img.height() * bpp);
    for (size_t y=0; y<img.height(); ++y)
    {
        for (size_t x=0; x<img.width(); ++x)
        {
            reference_image_std2[4*((img.height() - y - 1)*img.width() + x)    ] = img.data()[4*(y*img.width() + x)];
            reference_image_std2[4*((img.height() - y - 1)*img.width() + x) + 1] = img.data()[4*(y*img.width() + x) + 1];
            reference_image_std2[4*((img.height() - y - 1)*img.width() + x) + 2] = img.data()[4*(y*img.width() + x) + 2];
            reference_image_std2[4*((img.height() - y - 1)*img.width() + x) + 3] = img.data()[4*(y*img.width() + x) + 3];
        }
    }
    //memcpy(reference_image_std2.data(), img.data(), img.width() * img.height() * bpp);
    reference_image = cv::Mat(img.height(), img.width(), CV_8UC4, reinterpret_cast<void*>(reference_image_std2.data()));

    reference_keypoints.clear();
    orb->detectAndCompute(reference_image, cv::noArray(), reference_keypoints, reference_descriptors);
    std::cout << "Found " << reference_descriptors.size() << " descriptors.\n";

    matcher->clear();
    matcher->add(reference_descriptors);

    matcher_initialized = true;
}

void renderer::update_reference_image()
{
    reference_image_std = get_current_image();
    reference_image = cv::Mat(rt.height(), rt.width(), CV_8UC4, reinterpret_cast<void*>(reference_image_std.data()));

    reference_keypoints.clear();
    orb->detectAndCompute(reference_image, cv::noArray(), reference_keypoints, reference_descriptors);
    std::cout << "Found " << reference_descriptors.size() << " descriptors.\n";

    matcher->clear();
    matcher->add(reference_descriptors);

    matcher_initialized = true;
}

match_result_t renderer::match()
{
    if (!matcher_initialized) return {};

    // get current image
    auto current_image_std = get_current_image();
    auto current_image = cv::Mat(rt.height(), rt.width(), CV_8UC4, reinterpret_cast<void*>(current_image_std.data()));

    // detect
    std::vector<cv::KeyPoint> current_keypoints;
    cv::Mat current_descriptors;
    orb->detectAndCompute(current_image, cv::noArray(), current_keypoints, current_descriptors);

    // match
    match_result_t result;
    matcher->match(current_descriptors, result.matches, cv::noArray());
    result.num_ref_descriptors = reference_descriptors.size().height;

//    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
//    cv::Mat img;
//    cv::drawMatches(current_image, current_keypoints, reference_image, reference_keypoints, result.matches, img);
//    cv::imshow("Display Image", img);
//    cv::waitKey(0);
    return result;
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
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    rend.load_volume();
    rend.load_reference_image();

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
