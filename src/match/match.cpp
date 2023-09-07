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
#include <opencv2/cudaimgproc.hpp> // cvt::cuda::cvtColor
#include <opencv2/imgproc.hpp> // cv::cvtColor
#include <opencv2/cudafeatures2d.hpp> // cv::cuda::ORB
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
#include "orb_matcher.h"
#include "render.h"
#include "timer.h"

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
        , matcher()
        , selected_point(0)
        , selected_pixels()
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

    // volume rendering
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
    virvo::PixelFormat                                  texture_format;
    float                                               delta;
    float                                               integration_coefficient;
    vec3                                                bgcolor;
    vec2f                                               value_range;
    // matcher
    orb_matcher                                         matcher;
    // pixel select
    int                                                 selected_point;
    vec2                                                selected_pixels[4];
    pinhole_camera                                      saved_cameras[2];
    ray_type_cpu                                        saved_rays[4];

    void load_volume();
    void load_reference_image();
    void update_reference_image();
    void update_reference_image(const cv::Mat& image);
    match_result_t match();
    void search();
    void search2();
    void search_impl(const search_mode mode, const int grid_size, const float search_distance);
    void search_impl_up(const float rotation_range);
    std::vector<vector<4, unorm<8>>> get_current_image();
    std::pair<vec3, vec3> find_closest_points(ray_type_cpu r1, ray_type_cpu r2);
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

    // enable overlay when in selection mode
    if (selected_point > 0)
    {
        glEnable(GL_SCISSOR_TEST);
        const auto& pix = selected_pixels[selected_point - 1];
        constexpr int dot_size = 5;
        constexpr int offset = dot_size / 2;
        glScissor(pix.x - offset, pix.y - offset, dot_size, dot_size);
        const vec3f color1{1.0, 0.0, 0.0};
        const vec3f color2{0.0, 1.0, 0.0};
        const auto& color = selected_point > 2 ? color1 : color2;
        glClearColor(color.x, color.y, color.z, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_SCISSOR_TEST);
    }
}

void renderer::search()
{
    auto match_result = match();
    auto good_matches = match_result.good_matches();
    std::vector<cv::Point2f> reference_points;
    std::vector<cv::Point2f> query_points;
    for (auto& m : good_matches)
    {
        reference_points.push_back(match_result.reference_keypoints[m.trainIdx].pt);
        query_points.push_back(match_result.query_keypoints[m.queryIdx].pt);
    }

    //auto homography = cv::findHomography(query_points, reference_points, cv::RANSAC);

    // distance estimation: assume (tnear+tfar)/2 for now
    std::vector<cv::Point3f> reference_coords;
    auto camera = cam;
    const auto viewport = camera.get_viewport();
    camera.begin_frame();
    for (auto& p : reference_points)
    {
        auto r = camera.primary_ray(ray_type_cpu(), p.x, p.y, (float)viewport.w, (float)viewport.h);
        auto hr = intersect(r, bbox);
        auto coord = r.ori + r.dir * (hr.tnear + hr.tfar) / 2.f;
        reference_coords.push_back({coord.x, coord.y, coord.z});
    }
    camera.end_frame();

    // camera calibration
    double fx = 0.5 * ((double)viewport.w - 1) / std::tan(0.5 * camera.fovy() * camera.aspect());
    double fy = 0.5 * ((double)viewport.h - 1) / std::tan(0.5 * camera.fovy());
    double cx = ((double)viewport.w - 1) / 2.0;
    double cy = ((double)viewport.h - 1) / 2.0;
    // opencv stores in row-major order
    double camera_matrix_data[] = {fx, 0, cx, 0, fy, cy, 0, 0, 1};
    cv::Mat camera_matrix = cv::Mat(3, 3, CV_64F, camera_matrix_data);
    //camera_matrix = camera_matrix.t();

    // solve
    cv::Mat rotation, translation;
    std::cout << reference_coords.size() << ", " << query_points.size() << std::endl;
    cv::solvePnPRansac(reference_coords, query_points, camera_matrix, {}, rotation, translation);

    // get position
    cv::Mat rotation_matrix;
    cv::Rodrigues(rotation, rotation_matrix);
    cv::Mat camera_position = - rotation_matrix.t() * translation;
    // flip z axis
    camera_position.at<double>(2) *= -1.0;

    std::cout << "camera.eye() = " << camera.eye() << "\n";
    std::cout << "camera_position = \n" << camera_position << "\n";
}

void renderer::search2()
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
        timer t;
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
        std::cout << t.elapsed() << "ms\n";
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

    //float progress = 0;
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
//            matcher.mode = orb_matcher::matcher_mode::GPU;
        }
        else
        {
            std::cout << "Switching to CPU.\n";
            rt.mode() = host_device_rt::CPU;
//            matcher.mode = orb_matcher::matcher_mode::CPU;
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

    case keyboard::key::F1:
        std::cout << "Saved current cam 1." << "\n";
        saved_cameras[0] = cam;
        break;

    case keyboard::key::F2:
        std::cout << "Saved current cam 2." << "\n";
        saved_cameras[1] = cam;
        break;

    case keyboard::key::One:
        selected_point = 1;
        std::cout << "Pixel selection for point " << selected_point << "\n";
        break;

    case keyboard::key::Two:
        selected_point = 2;
        std::cout << "Pixel selection for point " << selected_point << "\n";
        break;

    case keyboard::key::Three:
        selected_point = 3;
        std::cout << "Pixel selection for point " << selected_point << "\n";
        break;

    case keyboard::key::Four:
        selected_point = 4;
        std::cout << "Pixel selection for point " << selected_point << "\n";
        break;

    case keyboard::key::ArrowUp:
        if (selected_point > 0)
        {
            const int offset = event.modifiers() == 0x00000004 ? 20 : 1;
            if (selected_pixels[selected_point - 1].y + offset < rt.height())
            {
                ++selected_pixels[selected_point - 1].y += offset;
            }
        }
        break;

    case keyboard::key::ArrowDown:
        if (selected_point > 0)
        {
            const int offset = event.modifiers() == 0x00000004 ? 20 : 1;
            if (selected_pixels[selected_point - 1].y - offset >= 0)
            {
                --selected_pixels[selected_point - 1].y -= offset;
            }
        }
        break;

    case keyboard::key::ArrowLeft:
        if (selected_point > 0)
        {
            const int offset = event.modifiers() == 0x00000004 ? 20 : 1;
            if (selected_pixels[selected_point - 1].x - offset >= 0)
            {
                --selected_pixels[selected_point - 1].x -= offset;
            }
        }
        break;

    case keyboard::key::ArrowRight:
        if (selected_point > 0)
        {
            const int offset = event.modifiers() == 0x00000004 ? 20 : 1;
            if (selected_pixels[selected_point - 1].x + offset < rt.width())
            {
                selected_pixels[selected_point - 1].x += offset;
            }
        }
        break;

    case keyboard::key::Enter:
        if (selected_point > 0)
        {
            std::cout << "Selected point " << selected_point << ": " << selected_pixels[selected_point - 1] << "\n";
            auto c = saved_cameras[(selected_point - 1) / 2];
            auto viewport = c.get_viewport();
            auto pix = selected_pixels[selected_point - 1];
            c.begin_frame();
            saved_rays[selected_point - 1] = c.primary_ray(ray_type_cpu(), pix.x, pix.y, (float)viewport.w, (float)viewport.h);
            c.end_frame();
            std::cout << "Saved ray " << selected_point
                << ": ori" << saved_rays[selected_point - 1].ori
                << " dir"  << saved_rays[selected_point - 1].dir << "\n";
            selected_point = 0;
        }
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
    int width, height;
    if (matcher.load_reference_image(reference_image_filename, width, height))
    {
        on_resize(width, height);
    }
}

void renderer::update_reference_image()
{
    auto reference_image_std = get_current_image();
    const auto reference_image = cv::Mat(rt.height(), rt.width(), CV_8UC4, reinterpret_cast<void*>(reference_image_std.data()));
    update_reference_image(reference_image);
}

void renderer::update_reference_image(const cv::Mat& image)
{
    matcher.init(image);
}

match_result_t renderer::match()
{
    // get current image
    auto current_image_std = get_current_image();
    auto current_image = cv::Mat(rt.height(), rt.width(), CV_8UC4, reinterpret_cast<void*>(current_image_std.data()));

    return matcher.match(current_image);
}

std::pair<vec3, vec3> renderer::find_closest_points(ray_type_cpu r1, ray_type_cpu r2)
{
    vec3 n  = cross(r1.dir, r2.dir);
    vec3 n1 = cross(r1.dir, n);
    vec3 n2 = cross(r2.dir, n);
    vec3 p1 = (dot(r1.ori, n1) - dot(r2.ori, n1)) / dot(r2.dir, n1) * r2.dir + r2.ori;
    vec3 p2 = (dot(r2.ori, n2) - dot(r1.ori, n2)) / dot(r1.dir, n2) * r1.dir + r1.ori;
    return std::make_pair(p1, p2);
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

//    rend.matcher.mode = (rend.rt.mode() == host_device_rt::CPU) ?
//                         orb_matcher::matcher_mode::CPU : orb_matcher::matcher_mode::GPU;
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
