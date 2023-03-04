// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <common/config.h>

#include <exception>
#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

#include <GL/glew.h>

#if VSNRAY_COMMON_HAVE_CUDA
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#endif

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/detail/platform.h>
#include <visionaray/math/math.h>
#include <visionaray/texture/texture.h>
#include <visionaray/pinhole_camera.h>
#include <visionaray/scheduler.h>

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/viewer_glut.h>

#include "fileio.h"
#include "host_device_rt.h"
#include "render.h"

using namespace visionaray;

using viewer_type   = viewer_glut;


//-------------------------------------------------------------------------------------------------
// Texture data
//

// volume data
VSNRAY_ALIGN(32) static const float voldata[2 * 2 * 2] = {

        // slice 1
        1.0f, 0.0f,
        0.0f, 1.0f,

        // slice 2
        0.0f, 1.0f,
        1.0f, 0.0f

        };

// post-classification transfer function
VSNRAY_ALIGN(32) static const vec4 tfdata[4 * 4] = {
        { 0.0f, 0.0f, 0.0f, 0.02f },
        { 0.7f, 0.1f, 0.2f, 0.03f },
        { 0.1f, 0.9f, 0.3f, 0.04f },
        { 1.0f, 1.0f, 1.0f, 0.05f }
        };


//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer : viewer_type
{
    renderer()
        : viewer_type(512, 512, "Visionaray Volume Rendering Example")
        , mode(projection_mode::AlphaCompositing)
        , bbox({ -1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f, 1.0f })
        , rt(
            host_device_rt::CPU,
            true /* double buffering */,
            false /* direct rendering */,
            host_device_rt::SRGB
            )
        , host_sched(8)
#if VSNRAY_COMMON_HAVE_CUDA
        , device_sched(8, 8)
#endif
        , volume_ref({std::array<unsigned int, 3>({2, 2, 2})})
        , transfunc_ref(4)
        , filename()
    {
        // Add cmdline options
        add_cmdline_option( support::cl::makeOption<std::string&>(
                    support::cl::Parser<>(),
                    "filename",
                    support::cl::Desc("Input file in nii format"),
                    support::cl::Positional,
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

        volume_ref.reset(voldata);
        volume_ref.set_filter_mode(Nearest);
        volume_ref.set_address_mode(Clamp);

        transfunc_ref.reset(tfdata);
        transfunc_ref.set_filter_mode(Linear);
        transfunc_ref.set_address_mode(Clamp);
#if VSNRAY_COMMON_HAVE_CUDA
        device_volume = cuda_volume_t(volume_ref);
        device_volume_ref = cuda_volume_ref_t(device_volume);
        device_transfunc = cuda_transfunc_t(transfunc_ref);
        device_transfunc_ref = cuda_transfunc_ref_t(device_transfunc);
#endif
    }

    projection_mode                                     mode;
    aabb                                                bbox;
    pinhole_camera                                      cam;
    host_device_rt                                      rt;
    tiled_sched<ray_type_cpu>                           host_sched;
    volume_ref_t                                        volume_ref;
    transfunc_ref_t                                     transfunc_ref;
#if VSNRAY_COMMON_HAVE_CUDA
    cuda_sched<ray_type_gpu>                            device_sched;
    cuda_volume_t                                       device_volume;
    cuda_volume_ref_t                                   device_volume_ref;
    cuda_transfunc_t                                    device_transfunc;
    cuda_transfunc_ref_t                                device_transfunc_ref;
#endif

    std::string                                         filename;
    fileio*                                             filereader;

    void load_volume();
protected:

    void on_display();
    void on_resize(int w, int h);

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
                transfunc_ref,
                bbox,
                rt,
                host_sched,
                cam,
                mode
            );

        // display the rendered image
        auto bgcolor = background_color();
        glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        rt.swap_buffers();
        rt.display_color_buffer();
    }
#if VSNRAY_COMMON_HAVE_CUDA
    else if (rt.mode() == host_device_rt::GPU)
    {
        render_cu(
                device_volume_ref,
                device_transfunc_ref,
                bbox,
                rt,
                device_sched,
                cam,
                mode
            );

        // display the rendered image
        auto bgcolor = background_color();
        glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        rt.swap_buffers();
        rt.display_color_buffer();
    }
#endif
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
// load volume if file
//
void renderer::load_volume()
{
    if (filename != "")
    {
        std::cout << "loading volume: " << filename << std::endl;
        filereader = new nifti_fileio();
        filereader->load(filename);

        auto dims = filereader->get_voxel_dimensions();
        std::array<unsigned int, 3> dims_array;
        dims_array[0] = dims.x;
        dims_array[1] = dims.y;
        dims_array[2] = dims.z;
        volume_ref = volume_ref_t(dims_array);
        volume_ref.reset(reinterpret_cast<const float*>(filereader->get_data()));
        transfunc_ref.reset(tfdata);
#if VSNRAY_COMMON_HAVE_CUDA
        device_volume = cuda_volume_t(volume_ref);
        device_volume_ref = cuda_volume_ref_t(device_volume);
        device_transfunc = cuda_transfunc_t(transfunc_ref);
        device_transfunc_ref = cuda_transfunc_ref_t(device_transfunc);
#endif
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
