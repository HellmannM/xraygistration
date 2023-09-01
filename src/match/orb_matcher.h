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


// Switch between CPU and GPU matcher
//TODO GPU matcher broken? Finds more matches than there are train descriptors?!
#define USE_GPU_MATCHER 0

struct orb_matcher
{
    enum matcher_mode {CPU=0, GPU};

    cv::Ptr<cv::ORB>                                    cpu_orb;
    cv::Ptr<cv::BFMatcher>                              cpu_matcher;
    bool                                                cpu_matcher_initialized;
    cv::Mat                                             cpu_reference_descriptors;
#if VSNRAY_COMMON_HAVE_CUDA
    cv::Ptr<cv::cuda::ORB>                              gpu_orb;
    cv::Ptr<cv::cuda::DescriptorMatcher>                gpu_matcher;
    bool                                                gpu_matcher_initialized;
    cv::cuda::GpuMat                                    gpu_reference_descriptors;
#endif
    std::vector<cv::KeyPoint>                           reference_keypoints;
    matcher_mode                                        mode;

    orb_matcher()
        : cpu_orb(cv::ORB::create(                                  // default values
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
        , cpu_matcher(cv::BFMatcher::create(cv::NORM_HAMMING, true))
        , cpu_matcher_initialized(false)
        , cpu_reference_descriptors()
        , gpu_orb(cv::cuda::ORB::create())
        , gpu_matcher(cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING))
        , gpu_matcher_initialized(false)
        , gpu_reference_descriptors()
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
            cpu_orb->detectAndCompute(reference_image, cv::noArray(), reference_keypoints, cpu_reference_descriptors);
            cpu_matcher->clear();
            cpu_matcher->add(cpu_reference_descriptors);
            cpu_matcher_initialized = true;
            std::cout << "Found " << cpu_reference_descriptors.size() << " descriptors.\n";
        }
    #if VSNRAY_COMMON_HAVE_CUDA
        else if (mode == matcher_mode::GPU)
        {
            cv::cuda::GpuMat gpu_reference_image_color(reference_image);
            cv::cuda::GpuMat gpu_reference_image;
            cv::cuda::cvtColor(gpu_reference_image_color, gpu_reference_image, cv::COLOR_RGBA2GRAY);
            gpu_orb->detectAndCompute(gpu_reference_image, cv::noArray(), reference_keypoints, gpu_reference_descriptors);
            #if USE_GPU_MATCHER
                gpu_matcher->clear();
                gpu_matcher->add({gpu_reference_descriptors});
                gpu_matcher_initialized = true;
            #else
                cpu_matcher->clear();
                gpu_reference_descriptors.download(cpu_reference_descriptors);
                cpu_matcher->add(cpu_reference_descriptors);
                cpu_matcher_initialized = true;
            #endif
            std::cout << "Found " << gpu_reference_descriptors.size() << " descriptors.\n";
        }
    #endif
    }
    
    match_result_t match(const cv::Mat& current_image)
    {
        std::vector<cv::KeyPoint> current_keypoints;
        match_result_t result;
    
        if (mode == matcher_mode::CPU)
        {
            if (!cpu_matcher_initialized) return {};
            cv::Mat cpu_current_descriptors;
            cpu_orb->detectAndCompute(current_image, cv::noArray(), current_keypoints, cpu_current_descriptors);
            cpu_matcher->match(cpu_current_descriptors, result.matches, cv::noArray());
            result.num_ref_descriptors = cpu_reference_descriptors.size().height;
        }
    #if VSNRAY_COMMON_HAVE_CUDA
        else if (mode == matcher_mode::GPU)
        {
            cv::cuda::GpuMat gpu_current_descriptors;
            cv::cuda::GpuMat gpu_current_image_color(current_image);
            cv::cuda::GpuMat gpu_current_image;
            cv::cuda::cvtColor(gpu_current_image_color, gpu_current_image, cv::COLOR_RGBA2GRAY);
            gpu_orb->detectAndCompute(gpu_current_image, cv::noArray(), current_keypoints, gpu_current_descriptors);
            #if USE_GPU_MATCHER
                if (!gpu_matcher_initialized) return {};
                gpu_matcher->match(gpu_current_descriptors, result.matches);
                result.num_ref_descriptors = gpu_reference_descriptors.size().height;
            #else
                if (!cpu_matcher_initialized) return {};
                cv::Mat cpu_current_descriptors;
                gpu_current_descriptors.download(cpu_current_descriptors);
                cpu_matcher->match(cpu_current_descriptors, result.matches, cv::noArray());
                result.num_ref_descriptors = cpu_reference_descriptors.size().height;
            #endif
        }
    #endif
        result.reference_keypoints = reference_keypoints;
        result.query_keypoints = current_keypoints;
    
    //    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    //    cv::Mat img;
    //    cv::drawMatches(current_image, current_keypoints, reference_image, reference_keypoints, result.matches, img);
    //    cv::imshow("Display Image", img);
    //    cv::waitKey(0);
        return result;
    }
};
