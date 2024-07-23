// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_VIEWER_RENDER_H
#define VSNRAY_VIEWER_RENDER_H 1

#include <common/config.h>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#endif

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

#include "host_device_rt.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Explicit template instantiation of render calls for faster parallel builds
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
// Helper types
//

using scalar_type_cpu           = float;
//using scalar_type_cpu           = simd::float4;
//using scalar_type_cpu           = simd::float8;
//using scalar_type_cpu           = simd::float16;
using scalar_type_gpu           = float;
using ray_type_cpu              = basic_ray<scalar_type_cpu>;
using ray_type_gpu              = basic_ray<scalar_type_gpu>;

using camera_t = pinhole_camera;
//using volume_value_t = unorm<16>;
//using volume_value_t = int16_t;
using volume_value_t = float;
using volume_t = texture<volume_value_t, 3>;
using transfunc_t = texture<vector<4, float>, 1>;
using volume_ref_t = volume_t::ref_type;
using transfunc_ref_t = transfunc_t::ref_type;
#ifdef __CUDACC__
using cuda_volume_t = cuda_texture<volume_value_t, 3>;
using cuda_transfunc_t = cuda_texture<vector<4, float>, 1>;
using cuda_volume_ref_t = cuda_volume_t::ref_type;
using cuda_transfunc_ref_t = cuda_transfunc_t::ref_type;
#endif

#if defined(__INTEL_COMPILER) || defined(__MINGW32__) || defined(__MINGW64__)
template <typename R>
using host_sched_t = tbb_sched<R>;
#else
template <typename R>
using host_sched_t = tiled_sched<R>;
#endif

void render_cpp(
        volume_ref_t const&         volume,
        aabb                        bbox,
        host_device_rt&             rt,
        host_sched_t<ray_type_cpu>& sched,
        camera_t const&             cam,
        float                       delta,
        float                       integration_coefficient
        );

float estimate_depth(
        volume_ref_t const& volume,
        aabb                bbox,
        basic_ray<float>    ray,
        float               delta,
        float               integration_coefficient,
        vec3f&              point
        );

#ifdef __CUDACC__
void render_cu(
        cuda_volume_ref_t const&    volume,
        aabb                        bbox,
        host_device_rt&             rt,
        cuda_sched<ray_type_gpu>&   sched,
        camera_t const&             cam,
        float                       delta,
        float                       integration_coefficient
        );
#endif

} // visionaray

#endif // VSNRAY_VIEWER_RENDER_H
