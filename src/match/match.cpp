// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <common/config.h>

#include <exception>
#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

#include <GL/glew.h>

//#if VSNRAY_COMMON_HAVE_CUDA
#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#endif

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
// volume data
VSNRAY_ALIGN(32) static const unorm<16> voldata_16ui[2 * 2 * 2] = {
//        65535,     0,
//        0    , 65535,
//        0    , 65535,
//        65535,     0
        1,     0,
        0    , 1,
        0    , 1,
        1,     0
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
//#if VSNRAY_COMMON_HAVE_CUDA
#ifdef __CUDACC__
        , device_sched(8, 8)
#endif
        , volume_ref({std::array<unsigned int, 3>({2, 2, 2})})
        , transfunc_ref(4)
        , filename()
        , texture_format(virvo::PF_R16UI)
    {
        // Add cmdline options
        add_cmdline_option( support::cl::makeOption<std::string&>(
                    support::cl::Parser<>(),
                    "filename",
                    support::cl::Desc("Input file in nii format"),
                    support::cl::Positional,
                    support::cl::init(filename)
                    ) );

//#if VSNRAY_COMMON_HAVE_CUDA
#ifdef __CUDACC__
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

    projection_mode                                     mode;
    aabb                                                bbox;
    pinhole_camera                                      cam;
    host_device_rt                                      rt;
    tiled_sched<ray_type_cpu>                           host_sched;
    volume_t                                            volume;
    volume_ref_t                                        volume_ref;
    transfunc_t                                         transfunc;
    transfunc_ref_t                                     transfunc_ref;
//#if VSNRAY_COMMON_HAVE_CUDA
#ifdef __CUDACC__
    cuda_sched<ray_type_gpu>                            device_sched;
    cuda_volume_t                                       device_volume;
    cuda_volume_ref_t                                   device_volume_ref;
    cuda_transfunc_t                                    device_transfunc;
    cuda_transfunc_ref_t                                device_transfunc_ref;
#endif

    std::string                                         filename;
    vvVolDesc*                                          vd;
    // Internal storage format for textures
    virvo::PixelFormat                                  texture_format;

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
//#if VSNRAY_COMMON_HAVE_CUDA
#ifdef __CUDACC__
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
    if (filename == "")
    {
        std::cerr << "No volume file provided. Using default volume." << std::endl;
        volume_ref.reset(voldata_16ui);
        volume_ref.set_filter_mode(Nearest);
        volume_ref.set_address_mode(Clamp);

        transfunc_ref.reset(tfdata);
        transfunc_ref.set_filter_mode(Linear);
        transfunc_ref.set_address_mode(Clamp);
#ifdef __CUDACC__
        device_volume_ref = cuda_volume_ref_t(device_volume);
        device_transfunc = cuda_transfunc_t(transfunc_ref);
        device_transfunc_ref = cuda_transfunc_ref_t(device_transfunc);
#endif
        return;
    }

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
    // Set default color scheme if no TF present:
    if (vd->tf[0].isEmpty())
    {
      vd->tf[0].setDefaultAlpha(0, 0.0, 1.0);
      vd->tf[0].setDefaultColors((vd->getChan()==1) ? 0 : 2, 0.0, 1.0);
    }
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
    volume.reset(reinterpret_cast<volume_ref_t::value_type const*>(tex_data));
    volume_ref = volume_ref_t(volume);
    volume_ref.set_address_mode(Clamp);
    volume_ref.set_filter_mode(Nearest);
    // update tf
    aligned_vector<vec4> tf(256 * 1 * 1);
    vd->computeTFTexture(0, 256, 1, 1, reinterpret_cast<float*>(tf.data()));
    transfunc = transfunc_t(tf.size());
    transfunc.reset(tf.data());
    transfunc_ref = transfunc_ref_t(transfunc);
    transfunc_ref.set_address_mode(Clamp);
    transfunc_ref.set_filter_mode(Nearest);
//#if VSNRAY_COMMON_HAVE_CUDA
#ifdef __CUDACC__
    device_volume = cuda_volume_t(volume_ref);
    device_volume_ref = cuda_volume_ref_t(device_volume);
    device_transfunc = cuda_transfunc_t(transfunc_ref);
    device_transfunc_ref = cuda_transfunc_ref_t(device_transfunc);
#endif

    //TODO update bbox
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
