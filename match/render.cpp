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
        float                           photon_energy
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

float estimate_depth(
        volume_ref_t const& volume,
        aabb                bbox,
        basic_ray<float>    ray,
        float               delta,
        float               photon_energy,
        vec3f&              point,
        float               contrib_epsilon_mm
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
    S accumulated_LAC = 0.0f;
    S max_value = 0.0f;
    S t_at_max_value = 0.0f;
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
        auto contribution = select(
                t < hit_rec.tfar,
                voxel,
                0.f);
        if (contribution > max_value)
        {
            max_value = contribution;
            t_at_max_value = t;
        }
        accumulated_LAC += contribution;
        // step on
        t += delta;
        ++steps;
    }

    auto average_LAC = accumulated_LAC / steps;
    auto traveled_distance_cm = (steps * delta) / S(10.0); // delta is in [mm]/[px]
    auto fraction_remaining = pow(photon_energy, -traveled_distance_cm * average_LAC);
    result.color = C(1.f) - C(fraction_remaining);

    // reject low attenuation pixels
    if (result.color < C(0.5f))
        return -1.f;

    point = ray.ori + ray.dir * t_at_max_value;

    // compare with epsilon around t_at_max_value
    const float start = max(t_at_max_value - contrib_epsilon_mm / 2.f, hit_rec.tnear);
    const float end   = min(t_at_max_value + contrib_epsilon_mm / 2.f, hit_rec.tfar);
    t = start;
    float section_accumulated_LAC = 0.0f;
    size_t section_steps = 0;
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
        section_accumulated_LAC += select(
                t < hit_rec.tfar,
                voxel,
                0.f);
        // step on
        t += delta;
        ++section_steps;
    }
    auto section_average_LAC = section_accumulated_LAC / section_steps;
    auto section_traveled_distance_cm = (section_steps * delta) / S(10.0); // delta is in [mm]/[px]
    auto section_fraction_remaining = pow(photon_energy, -section_traveled_distance_cm * section_average_LAC);
    
    // return percentage of magnitude contribution
    return (1.f - section_fraction_remaining) / (1.f - fraction_remaining);
}

} // visionaray
