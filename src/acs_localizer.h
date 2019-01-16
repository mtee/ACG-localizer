#define __STDC_LIMIT_MACROS

#include <vector>
#include <list>
#include <set>
#include <map>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <stdint.h>
#include <string>
#include <algorithm>
#include <climits>
#include <float.h>
#include <cmath>
#include <sstream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <queue>

#include <opencv2/core.hpp>
#include "features/visual_words_handler.hh"
// math functionality
#include "math/projmatrix.hh"
#include "math/matrix3x3.hh"

// tools to parse a bundler reconstruction
#include "sfm/parse_bundler.hh"
#include "sfm/bundler_camera.hh"

// RANSAC
//#include "RANSAC.hh"

// ANN Libary, used to perform search in 3D
#include <ANN/ANN.h>

namespace pca
{

// for unsigned char descriptors
class nearest_neighbors
{
  public:
    // constructors
    nearest_neighbors()
    {
        nn_idx1 = nn_idx2 = UINT32_MAX;
        dist1 = dist2 = -1;
    }

    nearest_neighbors(uint32_t nn1, uint32_t nn2, int d1, int d2) : nn_idx1(nn1), nn_idx2(nn2), dist1(d1), dist2(d2)
    {
    }

    nearest_neighbors(uint32_t nn1, int d1) : nn_idx1(nn1), nn_idx2(UINT32_MAX), dist1(d1), dist2(-1)
    {
    }

    nearest_neighbors(const nearest_neighbors &other)
    {
        if (&other != this)
        {
            nn_idx1 = other.nn_idx1;
            nn_idx2 = other.nn_idx2;
            dist1 = other.dist1;
            dist2 = other.dist2;
        }
    }

    // update the 2 nn with a new distance to a 3D points
    void update(uint32_t point, int dist)
    {
        if (dist1 < 0)
        {
            nn_idx1 = point;
            dist1 = dist;
        }
        else
        {
            if (dist < dist1)
            {
                nn_idx2 = nn_idx1;
                dist2 = dist1;
                nn_idx1 = point;
                dist1 = dist;
            }
            else if (dist < dist2 || dist2 < 0)
            {
                nn_idx2 = point;
                dist2 = dist;
            }
        }
    }

    float get_ratio()
    {
        return float(dist1) / float(dist2);
    }

    uint32_t nn_idx1, nn_idx2;
    int dist1, dist2;
};

// for the case that multiple descriptors of the same 3D point are mapped to the same visual word
class nearest_neighbors_multiple
{
  public:
    nearest_neighbors_multiple()
    {
        nn_idx1 = nn_idx2 = UINT32_MAX;
        dist1 = dist2 = -1;
    }

    nearest_neighbors_multiple(uint32_t nn1, uint32_t nn2, int d1, int d2) : nn_idx1(nn1), nn_idx2(nn2), dist1(d1), dist2(d2)
    {
    }

    nearest_neighbors_multiple(uint32_t nn1, int d1) : nn_idx1(nn1), nn_idx2(UINT32_MAX), dist1(d1), dist2(-1)
    {
    }

    nearest_neighbors_multiple(const nearest_neighbors_multiple &other)
    {
        if (&other != this)
        {
            nn_idx1 = other.nn_idx1;
            nn_idx2 = other.nn_idx2;
            dist1 = other.dist1;
            dist2 = other.dist2;
        }
    }

    void update(uint32_t point, int dist)
    {
        if (dist1 < 0)
        {
            nn_idx1 = point;
            dist1 = dist;
        }
        else
        {
            if (dist < dist1)
            {
                if (nn_idx1 != point)
                {
                    nn_idx2 = nn_idx1;
                    dist2 = dist1;
                }
                nn_idx1 = point;
                dist1 = dist;
            }
            else if ((dist < dist2 || dist2 < 0) && (point != nn_idx1))
            {
                nn_idx2 = point;
                dist2 = dist;
            }
        }
    }

    float get_ratio()
    {
        return float(dist1) / float(dist2);
    }

    uint32_t nn_idx1, nn_idx2;
    int dist1, dist2;
};

// structure to indicate a feature / point correspondence search is needed
// this is used for (feature, visual word) pairs as well as for 3D points that should be matched
// against the 2D features in the image
struct match_struct
{
    // id of the 2D feature, the 3D point
    uint32_t feature_id;

    // cost of the matching, i.e., the number of descriptors in a vw (2D-to-3D)
    // or the number of SIFT distances that have to be calculated for 3D-to-2D matching
    uint32_t matching_cost;

    // 2D-to-3D matching (true) or 3D-to-2D matching (false)
    bool matching_type;

    match_struct() : feature_id(0), matching_cost(0), matching_type(true) {}

    match_struct(uint32_t f, uint32_t c, bool t) : feature_id(f), matching_cost(c), matching_type(t) {}

    match_struct(const match_struct &other) : feature_id(other.feature_id), matching_cost(other.matching_cost), matching_type(other.matching_type) {}
};

class ACSLocalizer
{
  public:


    ACSLocalizer();
    cv::Mat processImage(cv::Mat img_gray_q, cv::Mat camMatrix, cv::Mat &inliers, std::vector<float> &c2D, std::vector<float> &c3D, cv::Mat &mDescriptors_q, std::set<size_t> &unique_vw);
    int init(std::string keylist, std::string bundle_file, uint32_t nb_clusters, std::string cluster_file, std::string vw_assignments, int prioritization_strategy, uint32_t _N_3D, int _consider_K_nearest_cams);
    void cleanUp();


  private:


    const uint64_t sift_dim = 128;
    const std::string outfile = "acg_results.txt";
    // First descriptor is stored in an array, while the second descriptor is stored in a vector (concatenation of vector entries)
    // The second descriptor begins at position index*128
    inline int
    compute_squared_SIFT_dist(cv::Mat v1, std::vector<unsigned char> &v2, uint32_t index)
    {
        uint64_t index_(index);
        index_ *= sift_dim;
        int dist = 0;
        int x = 0;
        for (uint64_t i = 0; i < sift_dim; ++i)
        {
            x = int(v1.at<float>(i)) - int(v2[index_ + i]);
            dist += x * x;
        }
        //  std::cout << "dist: " << dist << std::endl;
        return dist;
    }

    // generic comparison function, using < to compare the second entry of two pairs
    template <typename first_type, typename second_type>
    static inline bool cmp_second_entry_less(const std::pair<first_type, second_type> &a, const std::pair<first_type, second_type> &b)
    {
        return (a.second < b.second);
    }

    // comparison function for the new prioritization function
    static inline bool cmp_priorities(const match_struct &a, const match_struct &b)
    {
        return (a.matching_cost < b.matching_cost);
    }

    ////
    // constants
    ////

    // minimal number of inliers required to accept an image as registered
    uint32_t minimal_RANSAC_solution = 12;

    // SIFT-ratio value for 2D-to-3D matching. Since we store squared distances, we need the squared value 0.7^2 = 0.49
    float nn_ratio_2D_to_3D = 0.49f;

    // SIFT-ratio value for 2D-to-3D matching. Since we store squared distances, we need the squared value 0.6^2 = 0.36
    float nn_ratio_3D_to_2D = 0.36f;

    // the assumed minimal inlier ratio of RANSAC
    float min_inlier = 0.2f;

    // stop RANSAC if 60 seconds have passed
    double ransac_max_time = 60.0;

    // the number of nearest neighbors to search for in 3D
    int N_3D = 200;

    // compute a set cover from the images or use the original images
    bool use_image_set_cover = true;

    // the number of closest cameras to consider for clustering cameras
    uint32_t consider_K_nearest_cams = 10;

    // filter 3D points for visibiltiy before 3D-to-2D matching
    bool filter_points = false;

    // The number of correspondences to find before the search is terminated
    size_t max_cor_early_term = 100;

    //---------------------------------------------------------------------------------------------------------------------------------------------------------------
    // ACG DATA INITIALIZATION

    ////
    // load information about the 3D points:
    // the connected component it belongs to and the ids of the images it is observed in
    // we obtain all these information by parsing a Bundler file

    // for every 3D point, store the id of its connected component and the ids of the images that see the point
    std::vector<uint32_t> connected_component_id_per_point;
    std::vector<std::set<uint32_t>> images_per_point;

    std::vector<uint32_t> computed_visual_words;
    std::vector<uint32_t> computed_visual_words_low_dim;

    std::vector<double **> points_per_component;

    // store for every visual word a list of (point id, descriptor id) pairs
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> vw_points_descriptors;

    ////
    // load the visual words and their tree
    visual_words_handler vw_handler;

    std::vector<std::vector<uint32_t>> features_per_vw;

    // store for every visual word at the finest level the id of its parents at levels 2 and 3
    uint32_t *parents_at_level_2;
    uint32_t *parents_at_level_3;

    // for visual word, remember the number of points assigned to it
    std::vector<uint32_t> nb_points_per_vw;

    // store all descriptors in a single vector
    std::vector<unsigned char> all_descriptors;

    // for every connected component, get the number of points in it
    std::vector<uint32_t> nb_points_per_component;

    std::vector<std::vector<uint32_t>> indices_per_component;
    uint32_t nb_connected_components;

    ANNcoord **points3D = 0;
    std::vector<cv::Vec3b> colors_3D;
    std::vector<ANNkd_tree *> kd_trees;

    ANNkd_tree *kd_tree;

    ANNidxArray indices;
    ANNdistArray distances;

    cv::Mat camMatrix_raw = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat camMatrix_undistorted = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_64F);

    uint32_t nb_non_empty_vw, nb_3D_points, nb_descriptors;

    // for every point, store its descriptor ids and the ids of the visual words this descriptors belong to
    std::vector<std::vector<uint32_t>> desc_per_point;
    std::vector<std::vector<uint32_t>> vws_per_point;

    /*
*
*/
};
} // namespace pca