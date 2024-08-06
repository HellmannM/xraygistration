// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "render.h"

namespace visionaray
{

void render_cu(
        cuda_volume_ref_t const&        volume,
        aabb                            bbox,
        host_device_rt&                 rt,
        cuda_sched<ray_type_gpu>&       sched,
        camera_t const&                 cam,
        float                           delta,
        float                           photon_energy
        )
{
    auto sparams = make_sched_params(
            cam,
            rt
            );

    using R = ray_type_gpu;
    using S = R::scalar_type;
    using C = vector<4, S>;

    sched.frame([=] __device__ (R ray, int x, int y) -> result_record<S>
    {
        result_record<S> result;

        //bool debug = (x == 256) && (y == 256);
        //bool crosshair = (x == 256) || (y == 256);
        //if (crosshair) {result.color = C(1.f, 1.f, 1.f, 1.f); result.hit = true; return result;}

        auto hit_rec = intersect(ray, bbox);
        auto t = max(S(0.0), hit_rec.tnear);

        result.color = C(0.0);
        S accumulated_LAC = 0.0;
        size_t steps = 0;

        while ( any(t < hit_rec.tfar) )
        {
            auto pos = ray.ori + ray.dir * t;
            auto tex_coord = vector<3, S>(
                    ( pos.x + (bbox.size().x / 2) ) / bbox.size().x,
                    (-pos.y + (bbox.size().y / 2) ) / bbox.size().y,
                    (-pos.z + (bbox.size().z / 2) ) / bbox.size().z
                    );

            // sample volume
            auto voxel = tex3D(volume, tex_coord);
            accumulated_LAC += select(
                    t < hit_rec.tfar,
                    voxel,
                    0.f);

            // step on
            t += delta;
            ++steps;
        }

        auto average_LAC = accumulated_LAC / steps;
        auto traveled_distance_cm = (steps * delta) / S(10.0); // delta is in [mm]/[px]
        auto fraction_remaining = pow(photon_energy, -traveled_distance_cm * average_LAC);
        result.color = C(1.f) - C(fraction_remaining);

        result.hit = hit_rec.hit;
        return result;
    }, sparams);
}

} // visionaray
