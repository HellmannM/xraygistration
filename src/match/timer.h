#pragma once

#include <chrono>
#include <cstdint>

// simple timer
struct timer {
    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point last;

    timer()
    {
        start = std::chrono::system_clock::now();
        last  = start;
    }

    void reset()
    {
        start = std::chrono::system_clock::now();
        last  = start;
    }

    uint64_t elapsed()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - start).count();
    }

    uint64_t step()
    {
        uint64_t ret = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - last).count();
        last = std::chrono::system_clock::now();
        return ret;
    }
};
