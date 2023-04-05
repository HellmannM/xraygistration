// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "render.h"

namespace visionaray
{

void render_cpp(
        volume_ref_t const&             volume,
        aabb                            bbox,
        vec2f                           value_range,
        host_device_rt&                 rt,
        host_sched_t<ray_type_cpu>&     sched,
        camera_t const&                 cam,
        float                           delta
        )
{
    auto sparams = make_sched_params(
            cam,
            rt
            );

    using R = ray_type_cpu;
    using S = R::scalar_type;
    using C = vector<4, S>;

    sched.frame([&](R ray, int x, int y) -> result_record<S>
    {
        result_record<S> result;

        //bool debug = (x == 256) && (y == 256);
        //bool crosshair = (x == 256) || (y == 256);
        //if (crosshair) {result.color = C(1.f, 1.f, 1.f, 1.f); result.hit = true; return result;}

        auto hit_rec = intersect(ray, bbox);
        auto t = hit_rec.tnear;

        result.color = C(0.0);
        float line_integral = 0.0f;

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
            // clamp
            voxel = voxel < value_range.x ? value_range.x : voxel;
            //voxel = voxel > value_range.y ? value_range.y : voxel;
            line_integral += select(
                    t < hit_rec.tfar,
                    voxel - value_range.x,
                    0.f);

            // step on
            t += delta;
        }

        result.color = C(clamp(line_integral * delta * 0.000002f, 0.f, 1.f));

        result.hit = hit_rec.hit;
        return result;
    }, sparams);
}

} // visionaray
