// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <common/config.h>

#include <cstring> // memcpy
#include <exception>
#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

#include <boost/filesystem.hpp>

#include <GL/glew.h>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp> // cv::findChessboardCorners
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp> // cv::Mat
#include <opencv2/imgproc.hpp> // cv::cornerSubPix

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

#include <common/cpu_buffer_rt.h>
#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/viewer_glut.h>

#include "render.h"

using namespace visionaray;

using viewer_type   = viewer_glut;

//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer : viewer_type
{
    renderer()
        : viewer_type(1353, 1158, "Chess")
        , bbox({ -40.0f, -40.0f, 0.0f }, { 40.0f, 40.0f, -0.0001f })
        , host_sched(8)
    {
    }

    aabb                                                bbox;
    pinhole_camera                                      cam;
    cpu_buffer_rt<PF_RGBA8, PF_UNSPECIFIED, PF_RGBA32F> host_rt;
    tiled_sched<ray_type_cpu>                           host_sched;

    //std::vector<std::vector<unorm<8>>>                  saved_frames;                
    std::vector<std::vector<vector<4, unorm<8>>>>       saved_frames;                

    void screenshot();
    void calibrate();
    void reset();

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
    render_cpp(
            bbox,
            host_rt,
            host_sched,
            cam
        );

    // display the rendered image
    auto bgcolor = background_color();
    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    host_rt.display_color_buffer();
}


//-------------------------------------------------------------------------------------------------
// resize event
//

void renderer::on_resize(int w, int h)
{
    cam.set_viewport(0, 0, w, h);
    float aspect = w / static_cast<float>(h);
    cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    host_rt.resize(w, h);

    viewer_type::on_resize(w, h);
}

//-------------------------------------------------------------------------------------------------
// key press
//
void renderer::on_key_press(key_event const& event)
{
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wswitch"
    switch (event.key())
    {
    case 's':
        screenshot();
        break;
    case 'c':
        calibrate();
        break;
    case 'r':
        reset();
        break;
    }
    #pragma GCC diagnostic pop
    
    viewer_type::on_key_press(event);
}

void renderer::screenshot()
{
    std::vector<vector<4, unorm<8>>> rgba(host_rt.width() * host_rt.height());
    memcpy(
            rgba.data(),
            host_rt.color(),
            host_rt.width() * host_rt.height() * sizeof(vector<4, unorm<8>>)
    );

    // Flip so that origin is (top|left)
    std::vector<vector<4, unorm<8>>> flipped(host_rt.width() * host_rt.height());

    for (int y = 0; y < host_rt.height(); ++y)
    {
        for (int x = 0; x < host_rt.width(); ++x)
        {
            int yy = host_rt.height() - y - 1;
            flipped[yy * host_rt.width() + x] = rgba[y * host_rt.width() + x];
        }
    }

//    // Convert to greyscale
//    std::vector<unorm<8>> grey(host_rt.width() * host_rt.height());
//    for (int y = 0; y < host_rt.height(); ++y)
//    {
//        for (int x = 0; x < host_rt.width(); ++x)
//        {
//            auto rgba = flipped[y * host_rt.width() + x];
//            float premultiplied_pixel = ((float)rgba.x + (float)rgba.y + (float)rgba.z) / 4 * (float)rgba.w;
//            grey[y * host_rt.width() + x] = unorm<8>(premultiplied_pixel);
//        }
//    }

    saved_frames.push_back(flipped);
    std::cout << "saved " << saved_frames.size() << " frames.\n";
}

void renderer::calibrate()
{
    std::vector<cv::Vec3f> objects;
    for (int y=-30; y<=30; y+=10)
    {
        for (int x=-30; x<=30; x+=10)
        {
            objects.push_back(cv::Vec3f(x, y, 0.0f));
        }
    }

    std::vector<std::vector<cv::Vec3f>> objectPoints;
    std::vector<cv::Mat> cornerPoints;
    for (auto& frame : saved_frames)
    {
        auto frame_as_mat = cv::Mat(host_rt.height(), host_rt.width(), CV_8UC4, reinterpret_cast<void*>(frame.data()));
        cv::Mat frame_as_grey_mat;
        cv::cvtColor(frame_as_mat, frame_as_grey_mat, cv::COLOR_RGBA2GRAY);
        cv::Mat corners;
        if (cv::findChessboardCorners(frame_as_grey_mat, cv::Size(7, 7), corners))
        {
            cv::cornerSubPix(
                    frame_as_grey_mat,
                    corners,
                    cv::Size(5, 5),
                    cv::Size(-1, -1),
                    cv::TermCriteria(cv::TermCriteria::Type::EPS + cv::TermCriteria::Type::COUNT, 30, 0.1)
            );
            cornerPoints.push_back(corners);
            objectPoints.push_back(objects);
        }
    }

    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    cv::calibrateCamera(
            objectPoints,
            cornerPoints,
            cv::Size(host_rt.height(), host_rt.width()),
            cameraMatrix,
            distCoeffs,
            rvecs,
            tvecs
    );
    std::cout << "cameraMatrix=\n" << cameraMatrix << "\n";
    std::cout << "distCoeffs=\n" << distCoeffs << "\n";
    std::cout << "width=" << width() << "\n";
    std::cout << "height=" << height() << "\n";
    const auto aspect = width() / static_cast<float>(height());
    std::cout << "aspect=" << aspect << "\n";
    const auto fovx = 2.0 * std::atan(std::tan(cam.fovy() / 2.0) * aspect);
    const auto fovy = cam.fovy();
    std::cout << "fovx=" << fovx << "\n";
    std::cout << "fovy=" << fovy << "\n";
    std::cout << "fx=" << 0.5 * ((double)width()) / std::tan(0.5 * fovx) << "\n";
    std::cout << "fy=" << 0.5 * ((double)height()) / std::tan(0.5 * fovy) << "\n";
    std::cout << "for opencv: fx=fy=" << 0.5 * ((double)height()) / std::tan(0.5 * fovy) / aspect << "\n";
}

void renderer::reset()
{
    saved_frames.clear();
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
