// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "render.h"

namespace visionaray
{

void render_cpp(
        volume_ref_t const&             volume,
        transfunc_ref_t const&          transfunc,
        aabb                            bbox,
        host_device_rt&                 rt,
        host_sched_t<ray_type_cpu>&     sched,
        camera_t const&                 cam,
        projection_algo                 algo,
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

        //bool debug = (x == 280) && (y == 200);
        //bool debug = (x == 256) && (y == 256);
        bool crosshair = (x == 256) || (y == 256);
        if (crosshair) {result.color = C(1.f, 1.f, 1.f, 1.f); result.hit = true; return result;}

        auto hit_rec = intersect(ray, bbox);
        auto t = hit_rec.tnear;

        result.color = C(0.0);

        while ( any(t < hit_rec.tfar) )
        {
            auto pos = ray.ori + ray.dir * t;
            auto tex_coord = vector<3, S>(
                    ( pos.x + 1.0f ) / 2.0f,
                    (-pos.y + 1.0f ) / 2.0f,
                    (-pos.z + 1.0f ) / 2.0f
                    );

            // sample volume and do post-classification
            auto voxel = tex3D(volume, tex_coord);
            C color = tex1D(transfunc, voxel);

            // premultiplied alpha
            color.xyz() *= color.w;

            // compositing
            if (algo == projection_algo::AlphaCompositing)
            {
                result.color += select(
                        t < hit_rec.tfar,
                        color * (1.0f - result.color.w),
                        C(0.0)
                        );

                // early-ray termination - don't traverse w/o a contribution
                if ( all(result.color.w >= 0.999f) )
                {
                    break;
                }
            }
            else if (algo == projection_algo::MaxIntensity)
            {
                result.color = select(
                        t < hit_rec.tfar,
                        max(color, result.color),
                        result.color
                        );
            }
            else if (algo == projection_algo::MinIntensity)
            {
                result.color = select(
                        t < hit_rec.tfar,
                        min(color, result.color),
                        result.color
                        );
            }
            else if (algo == projection_algo::DRR)
            {
                result.color += select(
                        t < hit_rec.tfar,
                        color,
                        C(0.0)
                        );
            }

            // step on
            t += delta;
        }

        result.hit = hit_rec.hit;
        return result;
    }, sparams);
}

} // visionaray
