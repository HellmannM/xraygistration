#pragma once

#include <iostream>
#include <string>
#include <vector>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp> // cvt::cuda::cvtColor
#include <opencv2/imgproc.hpp> // cv::cvtColor
#include <opencv2/core/mat.hpp> // cv::Mat
#include <opencv2/features2d.hpp> // cv::SIFT
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp> // cv::xfeatures2d::SURF

// Visionaray includes
#undef MATH_NAMESPACE
#include <common/image.h>

#include "match_result.h"

struct surf_sift_matcher
{
    cv::Ptr<cv::xfeatures2d::SURF>                      detector_surf;
    cv::Ptr<cv::SIFT>                                   descriptor_sift;
    cv::Ptr<cv::BFMatcher>                              matcher;
    bool                                                matcher_initialized;
    cv::Mat                                             reference_descriptors;
    std::vector<cv::KeyPoint>                           reference_keypoints;

    surf_sift_matcher()
        : detector_surf(cv::xfeatures2d::SURF::create())
        , descriptor_sift(cv::SIFT::create())
        , matcher(cv::BFMatcher::create(cv::NORM_L2, true)) //NORM_HAMMING for ORB
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
        detector_surf->detect(reference_image, reference_keypoints, cv::noArray());
        descriptor_sift->compute(reference_image, reference_keypoints, reference_descriptors);
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
        detector_surf->detect(current_image, current_keypoints, cv::noArray());
        descriptor_sift->compute(current_image, current_keypoints, current_descriptors);
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
