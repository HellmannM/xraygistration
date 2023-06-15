#include <vector>

#include <opencv2/opencv.hpp>

struct match_result
{
    static constexpr float match_ratio_compare_offset {0.005f};
    static constexpr float good_match_threshold {70.f};

    uint32_t num_ref_descriptors;
    std::vector<cv::DMatch> matches;

    match_result() = default;

    match_result(const match_result& rhs)
    : num_ref_descriptors(rhs.num_ref_descriptors), matches(rhs.matches) {}

    void operator=(const match_result& rhs)
    {
        num_ref_descriptors = rhs.num_ref_descriptors;
        matches = rhs.matches;
    }

    // "smart" comparator: compare match_ratio if significantly different, otherwise compare good_distance (if match_ratio is similar).
    bool operator<(const match_result& rhs) const
    {
        if (match_ratio() + match_ratio_compare_offset < rhs.match_ratio())
            return true;
        else if (match_ratio() < rhs.match_ratio())
            return good_matches_distance() < rhs.good_matches_distance();
        else
            return false;
    }

    // "smart" comparator: compare match_ratio if significantly different, otherwise compare good_distance (if match_ratio is similar).
    bool operator>(const match_result& rhs) const
    {
        if (match_ratio() + match_ratio_compare_offset > rhs.match_ratio())
            return true;
        else if (match_ratio() > rhs.match_ratio())
            return good_matches_distance() > rhs.good_matches_distance();
        else
            return false;
    }

    float match_ratio() const
    {
        if (num_ref_descriptors == 0)
            return -1;
        return (float)matches.size() / num_ref_descriptors;
    }

    float distance() const
    {
        float dist = 0.f;
        for (auto& m : matches)
        {
            dist += m.distance;
        }
        return dist;
    }

    std::vector<cv::DMatch> good_matches(const float threshold = good_match_threshold) const
    {
        std::vector<cv::DMatch> gm;
        for (auto& m : matches)
        {
            if (m.distance < threshold)
                gm.push_back(m);
        }
        return gm;
    }

    float good_matches_ratio(const float threshold = good_match_threshold) const
    {
        if (num_ref_descriptors == 0)
            return -1;
        return (float)good_matches(threshold).size() / num_ref_descriptors;
    }

    float good_matches_distance(const float threshold = good_match_threshold) const
    {
        float dist = 0.f;
        for (auto& m : good_matches(threshold))
        {
            dist += m.distance;
        }
        return dist;
    }
};
