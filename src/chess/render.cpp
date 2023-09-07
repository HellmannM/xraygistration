// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "render.h"

namespace visionaray
{

void render_cpp(
        aabb                            bbox,
        cpu_buffer_rt<PF_RGBA8, PF_UNSPECIFIED, PF_RGBA32F>&
                                        rt,
        host_sched_t<ray_type_cpu>&     sched,
        camera_t const&                 cam
        )
{
    auto sparams = make_sched_params(
            cam,
            rt
            );

    using R = ray_type_cpu;
    using S = R::scalar_type;
    using C = vector<4, S>;

    sched.frame([&](R ray) -> result_record<S>
    {
        result_record<S> result;

        auto hit_rec = intersect(ray, bbox);
        auto t = hit_rec.tnear;

        result.color = C(0.0, 0.0, 1.0, 1.0);
        
        if (hit_rec.hit)
        {
            auto pos = ray.ori + ray.dir * t;
            if ((int)((pos.x + 40.f)/ 10) & 0x1)
            {
                if ((int)((pos.y + 40.f)/ 10) & 0x1)
                {
                    result.color = C(1.0);
                }
                else
                {
                    result.color = C(0.0, 0.0, 0.0, 1.0);
                }
            }
            else
            {
                if ((int)((pos.y + 40.f)/ 10) & 0x1)
                {
                    result.color = C(0.0, 0.0, 0.0, 1.0);
                }
                else
                {
                    result.color = C(1.0);
                }
            }
        }

        result.hit = hit_rec.hit;
        return result;
    }, sparams);
}

} // visionaray
