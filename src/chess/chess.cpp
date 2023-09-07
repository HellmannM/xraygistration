// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <common/config.h>

#include <exception>
#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

#include <boost/filesystem.hpp>

#include <GL/glew.h>

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/detail/platform.h>
#include <visionaray/math/math.h>
#include <visionaray/texture/texture.h>
#include <visionaray/pinhole_camera.h>
#include <visionaray/scheduler.h>

#include <common/image.h>
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
        : viewer_type(500, 384, "Chess")
        , bbox({ -40.0f, -40.0f, 0.0f }, { 40.0f, 40.0f, -0.0001f })
        , host_sched(8)
    {
    }

    aabb                                                bbox;
    pinhole_camera                                      cam;
    cpu_buffer_rt<PF_RGBA8, PF_UNSPECIFIED, PF_RGBA32F> host_rt;
    tiled_sched<ray_type_cpu>                           host_sched;

    void screenshot();

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
    }
    #pragma GCC diagnostic pop
    
    viewer_type::on_key_press(event);
}

void renderer::screenshot()
{
    std::string screenshot_file_base = "screenshot";
#if VSNRAY_COMMON_HAVE_PNG
    static const std::string screenshot_file_suffix = ".png";
    image::save_option opt1;
#else
    static const std::string screenshot_file_suffix = ".pnm";
    image::save_option opt1({"binary", true});
#endif

    // Swizzle to RGB8 for compatibility with pnm image
    std::vector<vector<3, unorm<8>>> rgb(host_rt.width() * host_rt.height());
    swizzle(
        rgb.data(),
        PF_RGB8,
        host_rt.color(),
        PF_RGBA32F,
        host_rt.width() * host_rt.height(),
        TruncateAlpha
        );

    //if (rt.color_space() == host_device_rt::SRGB)
    {
        for (int y = 0; y < host_rt.height(); ++y)
        {
            for (int x = 0; x < host_rt.width(); ++x)
            {
                auto& color = rgb[y * host_rt.width() + x];
                color.x = powf(color.x, 1 / 2.2f);
                color.y = powf(color.y, 1 / 2.2f);
                color.z = powf(color.z, 1 / 2.2f);
            }
        }
    }

    // Flip so that origin is (top|left)
    std::vector<vector<3, unorm<8>>> flipped(host_rt.width() * host_rt.height());

    for (int y = 0; y < host_rt.height(); ++y)
    {
        for (int x = 0; x < host_rt.width(); ++x)
        {
            int yy = host_rt.height() - y - 1;
            flipped[yy * host_rt.width() + x] = rgb[y * host_rt.width() + x];
        }
    }

    image img(
        host_rt.width(),
        host_rt.height(),
        PF_RGB8,
        reinterpret_cast<uint8_t const*>(flipped.data())
        );

    int inc = 0;
    std::string inc_str = "";

    std::string filename = screenshot_file_base + inc_str + screenshot_file_suffix;

    while (boost::filesystem::exists(filename))
    {
        ++inc;
        inc_str = std::to_string(inc);

        while (inc_str.length() < 4)
        {
            inc_str = std::string("0") + inc_str;
        }

        inc_str = std::string("-") + inc_str;

        filename = screenshot_file_base + inc_str + screenshot_file_suffix;
    }

    if (img.save(filename, {opt1}))
    {
        std::cout << "Screenshot saved to file: " << filename << '\n';
    }
    else
    {
        std::cerr << "Error saving screenshot to file: " << filename << '\n';
    }

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
