#pragma once

#include <iostream>
#include <string>
#include <vector>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp> // cvt::cuda::cvtColor
#include <opencv2/imgproc.hpp> // cv::cvtColor
#include <opencv2/core/mat.hpp> // cv::Mat
#include <opencv2/features2d.hpp> // cv::SIFT, cv::ORB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp> // cv::xfeatures2d::SURF
#if VSNRAY_COMMON_HAVE_CUDA
#include <opencv2/cudafeatures2d.hpp> // cv::cuda::ORB
#endif

// Visionaray includes
#undef MATH_NAMESPACE
#include <common/image.h>

#include "match_result.h"

namespace detector_type
{
    typedef cv::ORB ORB;
    typedef cv::xfeatures2d::SURF SURF;
#if VSNRAY_COMMON_HAVE_CUDA
    typedef cv::cuda::ORB ORB_GPU;
#endif
}
namespace descriptor_type
{
    typedef cv::ORB ORB;
    typedef cv::SIFT SIFT;
#if VSNRAY_COMMON_HAVE_CUDA
    typedef cv::cuda::ORB ORB_GPU;
#endif
}
namespace matcher_type
{
    typedef cv::BFMatcher BFMatcher;
#if VSNRAY_COMMON_HAVE_CUDA
    typedef cv::cuda::DescriptorMatcher BFMatcher_GPU;
#endif
}

//typedef cv::ORB matcher_type_ORB;
//typedef cv::SIFT matcher_type_SIFT;
//typedef cv::xfeatures2d::SURF matcher_type_SURF;
//#if VSNRAY_COMMON_HAVE_CUDA
//typedef cv::cuda::ORB matcher_type_ORB_GPU;
//#endif

template <typename Detector, typename Descriptor, typename Matcher>
struct feature_matcher
{
    cv::Ptr<Detector>           detector;
    cv::Ptr<Descriptor>         descriptor;
    cv::Ptr<Matcher>            matcher;
    bool                        matcher_initialized;
    cv::Mat                     reference_descriptors;
    std::vector<cv::KeyPoint>   reference_keypoints;
#if VSNRAY_COMMON_HAVE_CUDA
    cv::cuda::GpuMat            gpu_reference_descriptors;
#endif

    feature_matcher()
        : detector(Detector::create())
        , descriptor(Descriptor::create())
        , matcher(cv::BFMatcher::create(cv::NORM_L2, true))
        , matcher_initialized(false)
        , reference_descriptors()
        , reference_keypoints()
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
        detector->detect(reference_image, reference_keypoints, cv::noArray());
        descriptor->compute(reference_image, reference_keypoints, reference_descriptors);
        matcher->clear();
        matcher->add(reference_descriptors);
        matcher_initialized = true;
        std::cout << "Found " << reference_descriptors.size() << " descriptors.\n";
    }
    
    match_result_t match(const cv::Mat& current_image)
    {
        std::vector<cv::KeyPoint> current_keypoints;
        match_result_t result;
    
        if (!matcher_initialized) return {};
        cv::Mat current_descriptors;
        detector->detect(current_image, current_keypoints, cv::noArray());
        descriptor->compute(current_image, current_keypoints, current_descriptors);
        matcher->match(current_descriptors, result.matches, cv::noArray());
        result.num_ref_descriptors = reference_descriptors.size().height;
        result.reference_keypoints = reference_keypoints;
        result.query_keypoints = current_keypoints;
    
        //cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
        //cv::Mat img;
        ////cv::drawMatches(current_image, current_keypoints, reference_image, reference_keypoints, result.matches, img);
        //cv::drawMatches(current_image, current_keypoints, current_image, reference_keypoints, result.matches, img);
        //cv::imshow("Display Image", img);
        //cv::waitKey(0);
        return result;
    }
};

    template<>
    feature_matcher<cv::xfeatures2d::SURF, cv::SIFT, cv::BFMatcher>::feature_matcher()
        : detector(cv::xfeatures2d::SURF::create())
        , descriptor(cv::SIFT::create())
        , matcher(cv::BFMatcher::create(cv::NORM_L2, true))
        , matcher_initialized(false)
        , reference_descriptors()
        , reference_keypoints()
        {};

    template<>
    feature_matcher<cv::ORB, cv::ORB, cv::BFMatcher>::feature_matcher()
        : detector(cv::ORB::create(                             // default values
                /*int nfeatures     */ 5000,                    // 500
                /*float scaleFactor */ 1.1f,                    // 1.2f
                /*int nlevels       */ 15,                      // 8
                /*int edgeThreshold */ 10,                      // 31
                /*int firstLevel    */ 0,                       // 0
                /*int WTA_K         */ 2,                       // 2
                /*int scoreType     */ cv::ORB::HARRIS_SCORE,   // cv::ORB::HARRIS_SCORE
                /*int patchSize     */ 31,                      // 31
                /*int fastThreshold */ 10                       // 20
          ))
        , descriptor(cv::ORB::create())
        , matcher(cv::BFMatcher::create(cv::NORM_HAMMING, true))
        , matcher_initialized(false)
        , reference_descriptors()
        , reference_keypoints()
        {};

#if VSNRAY_COMMON_HAVE_CUDA
    template<>
    feature_matcher<cv::cuda::ORB, cv::cuda::ORB, cv::cuda::DescriptorMatcher>::feature_matcher()
        : detector(cv::cuda::ORB::create(                             // default values
                /*int nfeatures     */ 5000,                    // 500
                /*float scaleFactor */ 1.1f,                    // 1.2f
                /*int nlevels       */ 15,                      // 8
                /*int edgeThreshold */ 10,                      // 31
                /*int firstLevel    */ 0,                       // 0
                /*int WTA_K         */ 2,                       // 2
                /*int scoreType     */ cv::ORB::HARRIS_SCORE,   // cv::ORB::HARRIS_SCORE
                /*int patchSize     */ 31,                      // 31
                /*int fastThreshold */ 10                       // 20
          ))
        , descriptor(cv::cuda::ORB::create())
        , matcher(cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING))
        , matcher_initialized(false)
        , reference_descriptors()
        , reference_keypoints()
        {};

    template<>
    void feature_matcher<cv::cuda::ORB, cv::cuda::ORB, cv::cuda::DescriptorMatcher>::init(const cv::Mat& reference_image)
    {
        reference_keypoints.clear();
        cv::cuda::GpuMat gpu_reference_image_color(reference_image);
        cv::cuda::GpuMat gpu_reference_image;
        cv::cuda::cvtColor(gpu_reference_image_color, gpu_reference_image, cv::COLOR_RGBA2GRAY);
        detector->detectAndCompute(gpu_reference_image, cv::noArray(), reference_keypoints, gpu_reference_descriptors);
        matcher->clear();
        matcher->add({gpu_reference_descriptors});
        matcher_initialized = true;
        std::cout << "Found " << gpu_reference_descriptors.size() << " descriptors.\n";
    }

    template<>
    match_result_t feature_matcher<cv::cuda::ORB, cv::cuda::ORB, cv::cuda::DescriptorMatcher>::match(const cv::Mat& current_image)
    {
        std::vector<cv::KeyPoint> current_keypoints;
        match_result_t result;
        cv::cuda::GpuMat gpu_current_descriptors;
        cv::cuda::GpuMat gpu_current_image_color(current_image);
        cv::cuda::GpuMat gpu_current_image;
        cv::cuda::cvtColor(gpu_current_image_color, gpu_current_image, cv::COLOR_RGBA2GRAY);

        detector->detectAndCompute(gpu_current_image, cv::noArray(), current_keypoints, gpu_current_descriptors);
        if (!matcher_initialized) return {};
        matcher->match(gpu_current_descriptors, result.matches);

        result.num_ref_descriptors = gpu_reference_descriptors.size().height;
        result.reference_keypoints = reference_keypoints;
        result.query_keypoints = current_keypoints;
        return result;
    }
#endif
    
