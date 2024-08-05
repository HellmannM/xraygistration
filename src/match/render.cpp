// This file is distributed under the MIT license.
// See the LICENSE file for details.
#include <iostream>

#include "render.h"

namespace visionaray
{

void render_cpp(
        volume_ref_t const&             volume,
        aabb                            bbox,
        host_device_rt&                 rt,
        host_sched_t<ray_type_cpu>&     sched,
        camera_t const&                 cam,
        float                           delta,
        float                           integration_coefficient
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
        auto t = max(S(0.0), hit_rec.tnear);

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
            line_integral += select(
                    t < hit_rec.tfar,
                    voxel,
                    0.f);

            // step on
            t += delta;
        }

        constexpr float photon_energy = 13000.0;
        //TODO need traveled distance in cm
        float traveled_distance_cm = 0.01;
        float photon_energy_remaining = pow(photon_energy, -traveled_distance_cm * line_integral);
        //TODO inverse rescale photon_energy_remaining with photon_energy
        result.color = C(1.f) - C(clamp(photon_energy_remaining, 0.f, 1.f));

        result.hit = hit_rec.hit;
        return result;
    }, sparams);
}

float estimate_depth(
        volume_ref_t const& volume,
        aabb                bbox,
        basic_ray<float>    ray,
        float               delta,
        float               integration_coefficient,
        vec3f&              point
        )
{
    using R = ray_type_cpu;
    using S = R::scalar_type;
    using C = vector<4, S>;

    // trace ray again and get sum
    result_record<S> result;
    result.color = C(0.0);
    auto hit_rec = intersect(ray, bbox);
    auto t = max(S(0.0), hit_rec.tnear);
    float line_integral = 0.0f;
    float max_value = 0.0f;
    float t_max_value = 0.0f;
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
        auto contribution = select(
                t < hit_rec.tfar,
                voxel,
                0.f);
        if (contribution > max_value)
        {
            max_value = contribution;
            t_max_value = t;
        }
        line_integral += contribution;
        // step on
        t += delta;
    }
    constexpr float photon_energy = 13000.0;
    //TODO need traveled distance in cm
    float traveled_distance_cm = 0.01;
    float photon_energy_remaining = pow(photon_energy, -traveled_distance_cm * line_integral);
    //TODO inverse rescale photon_energy_remaining with photon_energy
    result.color = C(1.f) - C(clamp(photon_energy_remaining, 0.f, 1.f));
    // reject if not high density pixel
    if (result.color < C(0.5f))
        return -1.f;

    point = ray.ori + ray.dir * t_max_value;

    // add up epsilon around t_max and compare with line_integral.
    const auto bbox_dist = hit_rec.tfar - hit_rec.tnear;
    //TODO make search_dist a constant distance in the volume instead of relative to bbox intersectionts
    const auto search_dist = 0.2f * bbox_dist;
    const float start = t_max_value - search_dist / 2.f;
    const float end   = t_max_value + search_dist / 2.f;
    t = start;
    float sub_line_integral = 0.0f;
    while ( any(t < end) )
    {
        auto pos = ray.ori + ray.dir * t;
        auto tex_coord = vector<3, S>(
                ( pos.x + (bbox.size().x / 2) ) / bbox.size().x,
                (-pos.y + (bbox.size().y / 2) ) / bbox.size().y,
                (-pos.z + (bbox.size().z / 2) ) / bbox.size().z
                );
        // sample volume
        auto voxel = tex3D(volume, tex_coord);
        sub_line_integral += select(
                t < hit_rec.tfar,
                voxel,
                0.f);
        // step on
        t += delta;
    }
    
    // return percentage of magnitude contribution
    return sub_line_integral / line_integral;
}

} // visionaray
