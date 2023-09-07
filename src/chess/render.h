// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_VIEWER_RENDER_H
#define VSNRAY_VIEWER_RENDER_H 1

#include <common/config.h>

#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/math/simd/simd.h>
#include <visionaray/math/forward.h>
#include <visionaray/math/math.h>
#include <visionaray/math/ray.h>
#include <visionaray/math/vector.h>
#include <visionaray/aligned_vector.h>
#include <visionaray/environment_light.h>
#include <visionaray/generic_light.h>
#include <visionaray/pinhole_camera.h>
#include <visionaray/scheduler.h>

#if defined(__INTEL_COMPILER) || defined(__MINGW32__) || defined(__MINGW64__)
#include <visionaray/detail/tbb_sched.h>
#endif

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Explicit template instantiation of render calls for faster parallel builds
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
// Helper types
//


using scalar_type_cpu           = float;
using ray_type_cpu              = basic_ray<scalar_type_cpu>;

using camera_t = pinhole_camera;

#if defined(__INTEL_COMPILER) || defined(__MINGW32__) || defined(__MINGW64__)
template <typename R>
using host_sched_t = tbb_sched<R>;
#else
template <typename R>
using host_sched_t = tiled_sched<R>;
#endif

void render_cpp(
        aabb                            bbox,
        cpu_buffer_rt<PF_RGBA8, PF_UNSPECIFIED, PF_RGBA32F>&
                                        host_rt,
        host_sched_t<ray_type_cpu>&     sched,
        camera_t const&                 cam
        );


} // visionaray

#endif // VSNRAY_VIEWER_RENDER_H
