#pragma once

#include <iostream>
#include <string>
#include <vector>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp> // cvt::cuda::cvtColor
#include <opencv2/imgproc.hpp> // cv::cvtColor
#include <opencv2/cudafeatures2d.hpp> // cv::cuda::ORB
#include <opencv2/core/mat.hpp> // cv::Mat
#include <opencv2/features2d.hpp> // cv::ORB

// Visionaray includes
#undef MATH_NAMESPACE
#include <common/image.h>

#include "match_result.h"

struct orb_matcher
{
    enum matcher_mode {CPU=0, GPU};

    cv::Ptr<cv::ORB>                                    orb;
    cv::Ptr<cv::BFMatcher>                              matcher;
    bool                                                matcher_initialized;
    cv::Mat                                             reference_descriptors;
#if VSNRAY_COMMON_HAVE_CUDA
    cv::Ptr<cv::cuda::ORB>                              orb_gpu;
    cv::Ptr<cv::cuda::DescriptorMatcher>                matcher_gpu;
    bool                                                matcher_gpu_initialized;
    cv::cuda::GpuMat                                    reference_descriptors_gpu;
#endif
    std::vector<cv::KeyPoint>                           reference_keypoints;
    matcher_mode                                        mode;

    orb_matcher()
        : orb(cv::ORB::create(                                  // default values
                /*int nfeatures     */ 5000,                    // 500
                /*float scaleFactor */ 1.1f,                    // 1.2f
                /*int nlevels       */ 15,                       // 8
                /*int edgeThreshold */ 51,                      // 31
                /*int firstLevel    */ 0,                       // 0
                /*int WTA_K         */ 2,                       // 2
                /*int scoreType     */ cv::ORB::HARRIS_SCORE,   // cv::ORB::HARRIS_SCORE
                /*int patchSize     */ 51,                      // 31
                /*int fastThreshold */ 20                       // 20
          ))
        , matcher(cv::BFMatcher::create(cv::NORM_HAMMING, true))
        , matcher_initialized(false)
        , reference_descriptors()
        , orb_gpu(cv::cuda::ORB::create())
        , matcher_gpu(cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING))
        , matcher_gpu_initialized(false)
        , reference_descriptors_gpu()
        , reference_keypoints()
        , mode()
        {};

    bool load_reference_image(const std::string& reference_image_filename, int& width, int& height)
    {
        if (reference_image_filename.empty())
        {
            std::cout << "No reference image file provided.\n";
            return false;
        }
        std::cout << "Loading reference image file: " << reference_image_filename << "\n";
        visionaray::image img;
        img.load(reference_image_filename);
        std::cout << "width=" << img.width() << " height=" << img.height() << " pf=" << img.format() << "\n";
        width = img.width();
        height = img.height();
    
        int bpp = 4; //TODO
        std::vector<uint8_t> reference_image_std;
        reference_image_std.resize(img.width() * img.height() * bpp);
        for (size_t y=0; y<img.height(); ++y)
        {
            for (size_t x=0; x<img.width(); ++x)
            {
                reference_image_std[4*((img.height() - y - 1)*img.width() + x)    ] = img.data()[4*(y*img.width() + x)];
                reference_image_std[4*((img.height() - y - 1)*img.width() + x) + 1] = img.data()[4*(y*img.width() + x) + 1];
                reference_image_std[4*((img.height() - y - 1)*img.width() + x) + 2] = img.data()[4*(y*img.width() + x) + 2];
                reference_image_std[4*((img.height() - y - 1)*img.width() + x) + 3] = img.data()[4*(y*img.width() + x) + 3];
            }
        }
        //memcpy(reference_image_std.data(), img.data(), img.width() * img.height() * bpp);
        const auto reference_image = cv::Mat(img.height(), img.width(), CV_8UC4, reinterpret_cast<void*>(reference_image_std.data()));
    
        init(reference_image);

        return true;
    }
    
    void init(const cv::Mat& reference_image)
    {
        reference_keypoints.clear();
        if (mode == matcher_mode::CPU)
        {
            orb->detectAndCompute(reference_image, cv::noArray(), reference_keypoints, reference_descriptors);
            matcher->clear();
            matcher->add(reference_descriptors);
            matcher_initialized = true;
            std::cout << "Found " << reference_descriptors.size() << " descriptors.\n";
        }
    #if VSNRAY_COMMON_HAVE_CUDA
        else if (mode == matcher_mode::GPU)
        {
            cv::cuda::GpuMat reference_image_gpu_color(reference_image);
            cv::cuda::GpuMat reference_image_gpu;
            cv::cuda::cvtColor(reference_image_gpu_color, reference_image_gpu, cv::COLOR_RGBA2GRAY);
            orb_gpu->detectAndCompute(reference_image_gpu, cv::noArray(), reference_keypoints, reference_descriptors_gpu);
            matcher_gpu->clear();
            matcher_gpu->add({reference_descriptors_gpu});
            matcher_gpu_initialized = true;
            std::cout << "Found " << reference_descriptors_gpu.size() << " descriptors.\n";
        }
    #endif
    }
    
    match_result_t match(const cv::Mat& current_image)
    {
        std::vector<cv::KeyPoint> current_keypoints;
        match_result_t result;
    
        if (mode == matcher_mode::CPU)
        {
            if (!matcher_initialized) return {};
            cv::Mat current_descriptors;
            orb->detectAndCompute(current_image, cv::noArray(), current_keypoints, current_descriptors);
            matcher->match(current_descriptors, result.matches, cv::noArray());
            result.num_ref_descriptors = reference_descriptors.size().height;
        }
    #if VSNRAY_COMMON_HAVE_CUDA
        else if (mode == matcher_mode::GPU)
        {
            if (!matcher_gpu_initialized) return {};
            cv::cuda::GpuMat current_descriptors_gpu;
            cv::cuda::GpuMat current_image_gpu_color(current_image);
            cv::cuda::GpuMat current_image_gpu;
            cv::cuda::cvtColor(current_image_gpu_color, current_image_gpu, cv::COLOR_RGBA2GRAY);
            orb_gpu->detectAndCompute(current_image_gpu, cv::noArray(), current_keypoints, current_descriptors_gpu);
            matcher_gpu->match(current_descriptors_gpu, result.matches);
            result.num_ref_descriptors = reference_descriptors_gpu.size().height;
        }
    #endif
    
    //    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    //    cv::Mat img;
    //    cv::drawMatches(current_image, current_keypoints, reference_image, reference_keypoints, result.matches, img);
    //    cv::imshow("Display Image", img);
    //    cv::waitKey(0);
        return result;
    }
};
