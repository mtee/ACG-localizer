#define __STDC_LIMIT_MACROS


#include "acs_localizer.h"


// C++ includes
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
#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>

// includes for classes dealing with SIFT-features
#include "features/SIFT_loader.hh"
#include "features/visual_words_handler.hh"

// stopwatch
#include "timer.hh"

// math functionality
#include "math/projmatrix.hh"
#include "math/matrix3x3.hh"

// tools to parse a bundler reconstruction
#include "sfm/parse_bundler.hh"
#include "sfm/bundler_camera.hh"

// ANN Libary, used to perform search in 3D
#include <ANN/ANN.h>

namespace pca{




// check whether two sets have a common element or not
// sets are assumed to be stored in ascending order
// run time is in O( m + n ), where n and m are the sizes of the two sets
bool set_intersection_test(const std::set<uint32_t> &a, const std::set<uint32_t> &b)
{
    std::set<uint32_t>::const_iterator it_a = a.begin();
    std::set<uint32_t>::const_iterator it_b = b.begin();

    std::set<uint32_t>::const_iterator it_a_end = a.end();
    std::set<uint32_t>::const_iterator it_b_end = b.end();

    while ((it_a != it_a_end) && (it_b != it_b_end))
    {
        if (*it_a < *it_b)
            ++it_a;
        else if (*it_b < *it_a)
            ++it_b;
        else // equal, that is the set has a common element
            return true;
    }

    return false;
}


ACSLocalizer::ACSLocalizer(){
    computed_visual_words.resize(50000);
    std::fill(computed_visual_words.begin(), computed_visual_words.end(), 0);

    computed_visual_words_low_dim.resize(50000);
    std::fill(computed_visual_words_low_dim.begin(), computed_visual_words_low_dim.end(), 0);

    features_per_vw.resize(1000);
}


cv::Mat ACSLocalizer::processImage(cv::Mat img_gray_q, cv::Mat camMatrix, cv::Mat &inliers, std::vector<float> &c2D, std::vector<float> &c3D, cv::Mat &mDescriptors_q, std::set<size_t> &unique_vw)
{
    cv::Mat out = cv::Mat::zeros(4, 4, CV_64F);

    Timer featureTimer;
    featureTimer.Init();
    featureTimer.Start();

    cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> detector = cv::xfeatures2d::SiftFeatureDetector::create();

    std::vector<cv::KeyPoint> kps_q;

//    std::cout << "running sift detector on image size: " << img_gray_q.size() << " dim: " << img_gray_q.dims << " channels: " << img_gray_q.channels() << std::endl;
    detector->detectAndCompute(img_gray_q, cv::noArray(), kps_q, mDescriptors_q);
    featureTimer.Stop();
//    std::cout << "Feature extraction took " << featureTimer.GetElapsedTimeAsString() << " seconds" << std::endl;
    std::vector<SIFT_keypoint> keypoints;
    keypoints.resize(kps_q.size());
    for (int j = 0; j < mDescriptors_q.rows; j++)
    {
        SIFT_keypoint kp(kps_q[j].pt.x, kps_q[j].pt.y, kps_q[j].size, kps_q[j].angle * CV_PI / 180);
        keypoints[j] = kp;
    }

    uint32_t nb_loaded_keypoints = (uint32_t)keypoints.size();

    // center the keypoints around the center of the image
    // first we need to get the dimensions of the image
    int img_width = img_gray_q.cols;
    int img_height = img_gray_q.rows;
    //std::string jpg_filename( key_filenames[i] );

    for (uint32_t j = 0; j < nb_loaded_keypoints; ++j)
    {
        keypoints[j].x -= (img_gray_q.cols - 1.0) / 2.0f;
        keypoints[j].y = (img_gray_q.rows - 1.0) / 2.0f - keypoints[j].y;
    }

    std::cout << " loaded " << nb_loaded_keypoints << " descriptors" << std::endl;

    // assign the descriptors to the visual words
    Timer timer;
    timer.Init();
    timer.Start();
    computed_visual_words.clear();
    computed_visual_words_low_dim.clear();
    if (computed_visual_words.size() < nb_loaded_keypoints)
    {
        computed_visual_words.resize(nb_loaded_keypoints);
        computed_visual_words_low_dim.resize(nb_loaded_keypoints);
    }
    unique_vw.clear();
    vw_handler.set_nb_paths(1);
    vw_handler.assign_visual_words_ucharv(mDescriptors_q, nb_loaded_keypoints, computed_visual_words);
    timer.Stop();

    for (size_t j = 0; j < nb_loaded_keypoints; ++j)
        unique_vw.insert(computed_visual_words[j]);

//    std::cout << " assigned visual words in " << timer.GetElapsedTimeAsString() << " to " << unique_vw.size() << " unique vw" << std::endl;

    ////
    // establish 2D-3D correspondences by using the vw to compute pairwise nearest neighbors
    timer.Init();
    timer.Start();
    Timer all_timer;
    all_timer.Init();
    all_timer.Start();

    // first, compute for every feature in the image its visual word in the coarser vocabulary
    // compute the lower dimensions
    int low_dim_choosen = 0;
    int nb_low_dim = 100;

    if (nb_loaded_keypoints > 5000)
    {
        low_dim_choosen = 1;
        nb_low_dim = 1000;
    }

    features_per_vw.clear();
    features_per_vw.resize(1000);
    for (int j = 0; j < nb_low_dim; ++j)
        features_per_vw[j].clear();

    if (low_dim_choosen == 0)
    {
        for (size_t j = 0; j < nb_loaded_keypoints; ++j)
            features_per_vw[parents_at_level_2[computed_visual_words[j]]].push_back(j);
    }
    else
    {
        for (size_t j = 0; j < nb_loaded_keypoints; ++j)
            features_per_vw[parents_at_level_3[computed_visual_words[j]]].push_back(j);
    }
    c2D.clear();
    c3D.clear();

    std::vector<std::pair<uint32_t, uint32_t>> final_correspondences; // first the 2D, then the 3D point
    final_correspondences.clear();

    uint32_t max_corr = 0;
    uint32_t no_nn = 0;
    uint32_t no_sn_neighbor = 0;
    uint32_t failed_ratio = 0;

    // compute the priorities

    std::list<match_struct> priorities(nb_loaded_keypoints);
    std::list<match_struct>::iterator priorities_it = priorities.begin();

    for (uint32_t j = 0; j < nb_loaded_keypoints; ++j, ++priorities_it)
    {
        priorities_it->feature_id = j;
        priorities_it->matching_cost = nb_points_per_vw[computed_visual_words[j]];
        priorities_it->matching_type = true;
    }

    // keep track which 2D features are used for correspondences
    std::vector<bool> feature_in_correspondence(nb_loaded_keypoints, false);

    // keep track which 2D features are used for 3D-to-2D correspondences and
    // to which point they match. This is needed to be able to override 3D-to-2D
    // correspondences with new 3D-to-2D correspondences.
    // To signal that a feature is not part of a 3D-to-2D correspondence, we use
    // UINT32_MAX
    std::vector<uint32_t> point_per_feature(nb_loaded_keypoints, UINT32_MAX);

    // keep track which 3D points have been used in the query expansion
    std::set<uint32_t> used_3D_points;
    used_3D_points.clear();

    priorities.sort(ACSLocalizer::cmp_priorities);

    // similarly, we store for each 3D point the corresponding 2D feature as well as the squared distance
    std::map<uint32_t, std::pair<uint32_t, int>> corr_3D_to_2D;
    corr_3D_to_2D.clear();

    std::map<uint32_t, std::pair<uint32_t, int>>::iterator map_it_3D;

    // compute nearest neighbors using 2D-to-3D and 3D-to-2D matching

    uint32_t nb_considered_points = 0;
    uint32_t nb_considered_points_counter = 0;
    for (priorities_it = priorities.begin(); priorities_it != priorities.end(); ++priorities_it)
    {
        //         std::cout << priorities_it->feature_id << " " << priorities_it->matching_type << " " << priorities_it->matching_cost << std::endl;
        // check the matching type, and handle the different matching directions accordingly
        if (priorities_it->matching_type)
        {

            ////
            // 2D-to-3D matching, similar to the ICCV 2011 paper

            uint32_t j_index = priorities_it->feature_id;

            if (feature_in_correspondence[j_index])
                continue;

            uint32_t assignment = uint32_t(computed_visual_words[j_index]);

            ++nb_considered_points_counter;

            nearest_neighbors nn;

            if (priorities_it->matching_cost > 0)
            {
                // find nearest neighbor for 2D feature, update nearest neighbor information for 3D points if necessary
                size_t nb_poss_assignments = vw_points_descriptors[assignment].size();

                for (size_t k = 0; k < nb_poss_assignments; ++k)
                {
                    uint32_t point_id = vw_points_descriptors[assignment][k].first;
                    uint32_t desc_id = vw_points_descriptors[assignment][k].second;

                    int dist = compute_squared_SIFT_dist(mDescriptors_q.row(j_index), all_descriptors, desc_id);

                    nn.update(point_id, dist);
                }
            }

            // check if we have found a correspondence
            if (nn.dist1 >= 0)
            {
                if (nn.dist2 >= 0)
                {
                    if (nn.get_ratio() < nn_ratio_2D_to_3D)
                    {
                        // we found one, so we need check for mutual nearest neighbors
                        map_it_3D = corr_3D_to_2D.find(nn.nn_idx1);

                        if (map_it_3D != corr_3D_to_2D.end())
                        {
                            // a correspondence to the same 3D point already exists
                            // so we have to check whether we have to update it or not
                            if (map_it_3D->second.second > nn.dist1)
                            {
                                feature_in_correspondence[map_it_3D->second.first] = false;

                                map_it_3D->second.first = j_index;
                                map_it_3D->second.second = nn.dist1;

                                feature_in_correspondence[j_index] = true;
                            }
                        }
                        else
                        {
                            corr_3D_to_2D.insert(std::make_pair(nn.nn_idx1, std::make_pair(j_index, nn.dist1)));
                            feature_in_correspondence[j_index] = true;
                            used_3D_points.insert(nn.nn_idx1);

                            // avoid query expansion if we are not going to use it anyways
                            if (corr_3D_to_2D.size() >= max_cor_early_term)
                            {
                                nb_considered_points = nb_considered_points_counter;
                                break;
                            }

                            //// START ACTIVE SEARCH
                            // we only have to do the nn query in 3D space if we have
                            // found a new correspondence, not updated an old one (should be happening seldomly anyways)

                            // find the nearest neighbors in 3D
                            // ANN will throw an exception if we search for more points than contained in the connected component, so
                            // we have to adjust the number of points we search for
                            int N3D_ = std::min(N_3D, (int)nb_points_per_component[connected_component_id_per_point[nn.nn_idx1]]);
                            kd_trees[connected_component_id_per_point[nn.nn_idx1]]->annkSearch(points3D[nn.nn_idx1], N3D_, indices, distances);

                            ////
                            // find new matching possibilities and insert them into the correct position
                            // in the list
                            std::list<match_struct> new_matching_possibilities;
                            new_matching_possibilities.clear();

                            for (int kk = 0; kk < N3D_; ++kk)
                            {
                                if (indices[kk] < 0)
                                    break;

                                uint32_t candidate_point = indices_per_component[connected_component_id_per_point[nn.nn_idx1]][indices[kk]];

                                // check if we have already used or visited this 3D point (or promise to visit it later on
                                if (used_3D_points.find(candidate_point) == used_3D_points.end())
                                {
                                    // visibility filter
                                    if (filter_points && (!set_intersection_test(images_per_point[candidate_point], images_per_point[nn.nn_idx1])))
                                        continue;

                                    // promise that we will (eventually) look at this 3D point
                                    used_3D_points.insert(candidate_point);

                                    ////
                                    // compute the matching cost of this particular 3D point
                                    // therefore, we have to identify to which visual words (on a higher level)
                                    // the descriptors of this point are assigned how many times

                                    match_struct new_match(candidate_point, 0, false);

                                    if (low_dim_choosen == 0)
                                    {
                                        for (std::vector<uint32_t>::const_iterator it_vws = vws_per_point[candidate_point].begin(); it_vws != vws_per_point[candidate_point].end(); ++it_vws)
                                            new_match.matching_cost += features_per_vw[parents_at_level_2[*it_vws]].size();
                                    }
                                    else
                                    {
                                        for (std::vector<uint32_t>::const_iterator it_vws = vws_per_point[candidate_point].begin(); it_vws != vws_per_point[candidate_point].end(); ++it_vws)
                                            new_match.matching_cost += features_per_vw[parents_at_level_3[*it_vws]].size();
                                    }

                                    // push back the new matching possibilities
                                    new_matching_possibilities.push_back(new_match);
                                }
                            }

                            ////
                            // insert the list of new matching possibilities into our prioritization search structure
                            // to do so efficiently, we sort the new list and use insertion sort to update the prioritization scheme

                            // sort
                            new_matching_possibilities.sort(cmp_priorities);

                            // depending on the prioritization_strategy we either insert all new matching possibilities directly after the next
                            // entry in the current prioritization list (prioritization_strategy == 1) or use insertion sort to insert it in a
                            // sorted fashion (prioritization_strategy == 0)

                            // insertion sort: Notice that we start with the NEXT entry in the current prioritization list
                            // since we are nearly done with the old one
                            std::list<match_struct>::iterator insertion_it = priorities_it;
                            ++insertion_it;

                            std::list<match_struct>::iterator to_insert_it = new_matching_possibilities.begin();

                            while (to_insert_it != new_matching_possibilities.end())
                            {
                                if (insertion_it == priorities.end())
                                {
                                    priorities.insert(insertion_it, to_insert_it, new_matching_possibilities.end());
                                    break;
                                }

                                if (insertion_it->matching_cost > to_insert_it->matching_cost)
                                {
                                    priorities.insert(insertion_it, *to_insert_it);
                                    ++to_insert_it;
                                }
                                else
                                    ++insertion_it;
                            }

                            new_matching_possibilities.clear();

                            //// STOP ACTIVE SEARCH
                        }
                    }
                }
            }
        }
        else
        {
            ////
            // 3D-to-2D correspondence, handle it accordingly
            uint32_t candidate_point = priorities_it->feature_id;

            // check if we have already found a correspondence for that 3D point
            // if so, we don't need to find a new one since it was found during
            // 2D-to-3D matching which we trust more
            if (corr_3D_to_2D.find(candidate_point) != corr_3D_to_2D.end())
                continue;

            uint32_t nb_desc_for_point = (uint32_t)desc_per_point[candidate_point].size();
            std::vector<uint32_t> low_dim_vw_ids(nb_desc_for_point);

            // map all descriptors of that point to the lower dimensional descriptors
            std::set<uint32_t> low_dim_vw;
            low_dim_vw.clear();

            uint32_t counter = 0;
            if (low_dim_choosen == 0)
            {
                for (std::vector<uint32_t>::const_iterator it_vws = vws_per_point[candidate_point].begin(); it_vws != vws_per_point[candidate_point].end(); ++it_vws, ++counter)
                {
                    low_dim_vw_ids[counter] = parents_at_level_2[*it_vws];
                    low_dim_vw.insert(parents_at_level_2[*it_vws]);
                }
            }
            else
            {
                for (std::vector<uint32_t>::const_iterator it_vws = vws_per_point[candidate_point].begin(); it_vws != vws_per_point[candidate_point].end(); ++it_vws, ++counter)
                {
                    low_dim_vw_ids[counter] = parents_at_level_3[*it_vws];
                    low_dim_vw.insert(parents_at_level_3[*it_vws]);
                }
            }

            // try to find a new correspondence
            nearest_neighbors_multiple nn_exp;
            uint64_t descriptor_index(0);
            for (std::set<uint32_t>::const_iterator activated_vw = low_dim_vw.begin(); activated_vw != low_dim_vw.end(); ++activated_vw)
            {
                counter = 0;
                for (std::vector<uint32_t>::const_iterator it_desc = desc_per_point[candidate_point].begin(); it_desc != desc_per_point[candidate_point].end(); ++it_desc, ++counter)
                {
                    if (low_dim_vw_ids[counter] != *activated_vw)
                        continue;

                    descriptor_index = (*it_desc) * sift_dim;

                    for (std::vector<uint32_t>::const_iterator feature_it = features_per_vw[*activated_vw].begin(); feature_it != features_per_vw[*activated_vw].end(); ++feature_it)
                    {
                        int dist = 0;
                        int x = 0;
                        for (uint64_t jj = 0; jj < 128; ++jj)
                        {
                            x = ((int)mDescriptors_q.row(*feature_it).at<float>(jj)) - ((int)all_descriptors[descriptor_index + jj]);
                            dist += x * x;
                        }

                        nn_exp.update(*feature_it, dist);
                    }
                }
            }

            if (nn_exp.dist1 >= 0)
            {
                if (nn_exp.dist2 >= 0)
                {
                    if (nn_exp.get_ratio() < nn_ratio_3D_to_2D)
                    {
                        // we don't want to overwrite 2D-to-3D correspondence, only 3D-to-2D correspondence
                        if (!feature_in_correspondence[nn_exp.nn_idx1])
                        {
                            // no existing correspondence
                            corr_3D_to_2D.insert(std::make_pair(candidate_point, std::make_pair(nn_exp.nn_idx1, nn_exp.dist1)));
                            feature_in_correspondence[nn_exp.nn_idx1] = true;
                            point_per_feature[nn_exp.nn_idx1] = candidate_point;
                        }
                        else if (point_per_feature[nn_exp.nn_idx1] != UINT32_MAX)
                        {
                            // only 3D-to-2D correspondence
                            // overwrite if the absolute SIFT distance is smaller

                            // first, we to get an iterator to the corresponding 3D point
                            map_it_3D = corr_3D_to_2D.find(point_per_feature[nn_exp.nn_idx1]);

                            // this has to exist, otherwise we would not have labeled it as having a correspondence
                            if (map_it_3D->second.second > nn_exp.dist1)
                            {
                                // update the correspondence

                                // which means we first have to remove the old one
                                corr_3D_to_2D.erase(map_it_3D);

                                point_per_feature[nn_exp.nn_idx1] = candidate_point;

                                corr_3D_to_2D.insert(std::make_pair(candidate_point, std::make_pair(nn_exp.nn_idx1, nn_exp.dist1)));
                            }
                        }
                    }
                }
            }
        }

        //         std::cout << " "  << corr_3D_to_2D.size() << std::endl;

        if (corr_3D_to_2D.size() >= max_cor_early_term)
        {
            nb_considered_points = nb_considered_points_counter;
            break;
        }
    }

    if (nb_considered_points == 0)
        nb_considered_points = nb_loaded_keypoints;

    ////
    // Establish the correspondences needed for RANSAC-based pose estimation
        ////
        // Apply the RANSAC Pre-filter
        uint32_t max_set_size = 0;

        uint32_t nb_found_corr = (uint32_t)corr_3D_to_2D.size();

        // map found points to index range 0...N (actually the other direction)
        std::vector<uint32_t> index_to_point(nb_found_corr, 0);

        // go through all points found as correspondences, establish edges in the form of images
        std::map<uint32_t, std::list<uint32_t>> image_edges;
        image_edges.clear();

        uint32_t point_counter = 0;

        std::map<uint32_t, std::list<uint32_t>>::iterator it;

        for (map_it_3D = corr_3D_to_2D.begin(); map_it_3D != corr_3D_to_2D.end(); ++map_it_3D, ++point_counter)
        {

            index_to_point[point_counter] = map_it_3D->first;

            for (std::set<uint32_t>::const_iterator it_images_point = images_per_point[map_it_3D->first].begin(); it_images_point != images_per_point[map_it_3D->first].end(); ++it_images_point)
            {
                it = image_edges.find(*it_images_point);

                if (it != image_edges.end())
                    it->second.push_back(point_counter);
                else
                {
                    std::list<uint32_t> new_edge(1, point_counter);
                    std::pair<uint32_t, std::list<uint32_t>> p(*it_images_point, new_edge);
                    image_edges.insert(p);
                }
            }
        }

        ////
        // now find connected components, for every point, keep track which points belong to which connected component
        // also keep track of which images have been used

        std::vector<int> cc_per_corr(nb_found_corr, -1);
        int current_cc = -1;
        int max_cc = -1;
        uint32_t size_current_cc = 0;

        std::set<uint32_t>::const_iterator img_it;

        for (point_counter = 0; point_counter < nb_found_corr; ++point_counter)
        {
            if (cc_per_corr[point_counter] < 0)
            {
                // start new cc
                ++current_cc;
                size_current_cc = 0;

                std::queue<uint32_t> point_queue;
                point_queue.push(point_counter);

                // breadth first search for remaining points in connected component
                while (!point_queue.empty())
                {
                    uint32_t curr_point_id = point_queue.front();
                    point_queue.pop();

                    if (cc_per_corr[curr_point_id] < 0)
                    {
                        cc_per_corr[curr_point_id] = current_cc;
                        ++size_current_cc;

                        // add all points in images visible by this point
                        for (std::set<uint32_t>::const_iterator it_images_point = images_per_point[index_to_point[curr_point_id]].begin(); it_images_point != images_per_point[index_to_point[curr_point_id]].end(); ++it_images_point)
                        {
                            it = image_edges.find(*it_images_point);

                            for (std::list<uint32_t>::const_iterator p_it = it->second.begin(); p_it != it->second.end(); ++p_it)
                            {
                                if (cc_per_corr[*p_it] < 0)
                                    point_queue.push(*p_it);
                            }

                            // clear the image, we do not the this multi-edge anymore
                            it->second.clear();
                        }
                    }
                }

                if (size_current_cc > max_set_size)
                {
                    max_set_size = size_current_cc;
                    max_cc = current_cc;
                }
            }
        }

        ////
        // now generate the correspondences

        c2D.reserve(2 * max_set_size);
        c3D.reserve(3 * max_set_size);

        point_counter = 0;
        for (map_it_3D = corr_3D_to_2D.begin(); map_it_3D != corr_3D_to_2D.end(); ++map_it_3D, ++point_counter)
        {
            if (cc_per_corr[point_counter] == max_cc)
            {
                c2D.push_back(keypoints[map_it_3D->second.first].x);
                c2D.push_back(keypoints[map_it_3D->second.first].y);

                c3D.push_back(points3D[map_it_3D->first][0]);
                c3D.push_back(points3D[map_it_3D->first][1]);
                c3D.push_back(points3D[map_it_3D->first][2]);

                final_correspondences.push_back(std::make_pair(map_it_3D->second.first, map_it_3D->first));
            }
        }



    timer.Stop();
 //   std::cout << " computed correspondences in " << timer.GetElapsedTimeAsString() << ", considering " << nb_considered_points << " features "
 //             << " ( " << double(nb_considered_points) / double(nb_loaded_keypoints) * 100.0 << " % ) " << std::endl;

    ////
    // do the pose verification using RANSAC

 //   RANSAC::computation_type = P6pt;
//    RANSAC::stop_after_n_secs = true;
 //   RANSAC::max_time = ransac_max_time;
 //   RANSAC::error = 10.0f; // for P6pt this is the SQUARED reprojection error in pixels
//    RANSAC ransac_solver;

    uint32_t nb_corr = c2D.size() / 2;

    std::cout << " applying RANSAC on " << nb_corr << " correspondences out of " << corr_3D_to_2D.size() << std::endl;
    timer.Init();
    timer.Start();
    cv::Mat rvec = cv::Mat::zeros(1, 3, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(1, 3, CV_64F);
    std::vector<cv::Point3d> corr3d(nb_corr);

    corr3d.clear();
    for (int i = 0; i < c3D.size(); i += 3)
        corr3d.push_back(cv::Point3d(c3D[i], c3D[i + 1], c3D[i + 2]));

    std::vector<cv::Point2d> corr2d(nb_corr);
    corr2d.clear();
    for (int i = 0; i < c2D.size(); i += 2)
        corr2d.push_back(cv::Point2d(c2D[i] + (img_width - 1.0) / 2.0f, -c2D[i + 1] + (img_height - 1.0) / 2.0f));

    int inlierCount = 0;
    if (corr2d.size() > 4)
    {
        cv::solvePnPRansac(corr3d, corr2d, camMatrix, cv::Mat(), rvec, tvec, false, 500, 8.0F, 0.99, inliers, CV_EPNP);
        inlierCount = inliers.size().height;
        // std::cout << "opencv ransac pnp. Translation " << std::endl
        //           << tvec << std::endl
        //           << "inlier count: " << inlierCount << std::endl;

        if (inlierCount > 10)
        {

            cv::Mat sceneTransform = cv::Mat::eye(4, 4, CV_64F);
            cv::Rodrigues(rvec, sceneTransform.rowRange(0, 3).colRange(0, 3));
            sceneTransform.at<double>(0, 3) = tvec.at<double>(0, 0);
            sceneTransform.at<double>(1, 3) = tvec.at<double>(0, 1);
            sceneTransform.at<double>(2, 3) = tvec.at<double>(0, 2);

            sceneTransform.at<double>(3, 3) = 1;
            // TODO: move this code out
            // ofs << i << " " << inlierCount << " ";
            // for (int ti = 0; ti < 4; ti++)
            //     for (int tj = 0; tj < 4; tj++)
            //     {
            //         ofs << sceneTransform.at<double>(ti, tj) << " ";
            //     }
            // ofs << std::endl;
            //             pcm.AddOrUpdateFrustum("2", sceneTransform.inv(), 1, 1, 0, 0, 2);
            out = sceneTransform.inv();
        }
    }
    timer.Stop();
    all_timer.Stop();
    std::cout << "#########################" << std::endl;
    return out;
}

void ACSLocalizer::cleanUp()
{
    // delete kd-trees
    for (uint32_t i = 0; i < nb_connected_components; ++i)
    {
        for (uint32_t j = 0; j < nb_points_per_component[i]; ++j)
            points_per_component[i][j] = 0;

        delete[] points_per_component[i];
        points_per_component[i] = 0;
    }

    for (uint32_t i = 0; i < nb_3D_points; ++i)
    {
        if (points3D[i] != 0)
            delete[] points3D[i];
        points3D[i] = 0;
    }
    delete[] points3D;
    points3D = 0;

    for (uint32_t i = 0; i < nb_connected_components; ++i)
    {
        delete kd_trees[i];
        kd_trees[i] = 0;
    }

    delete kd_tree;
    kd_tree = 0;

    delete[] indices;
    delete[] distances;

    delete[] parents_at_level_2;
    parents_at_level_2 = 0;

    delete[] parents_at_level_3;
    parents_at_level_3 = 0;

    annClose();

    for (uint32_t i = 0; i < nb_3D_points; ++i)
        images_per_point[i].clear();
    images_per_point.clear();
}

int ACSLocalizer::init(std::string keylist, std::string bundle_file, uint32_t nb_clusters, std::string cluster_file, std::string vw_assignments, int prioritization_strategy, uint32_t _N_3D, int _consider_K_nearest_cams)
{

    computed_visual_words.resize(50000);
    std::fill(computed_visual_words.begin(), computed_visual_words.end(), 0);

    computed_visual_words_low_dim.resize(50000);
    std::fill(computed_visual_words_low_dim.begin(), computed_visual_words_low_dim.end(), 0);

    ////
    // for every visual word on level 2 or 3 (so at max 1000 visual words)
    // store the ids of the 2D features that are mapped to that visual word

    for (int i = 0; i < 1000; ++i)
        features_per_vw[i].resize(100);



    // rocketcluster
    if (prioritization_strategy != 0 && prioritization_strategy != 1 && prioritization_strategy != 2)
    {
        std::cerr << " ERROR: Unknown prioritization strategy " << prioritization_strategy << ", aborting" << std::endl;
        return -1;
    }
    std::cout << " Assumed minimal inlier-ratio: " << min_inlier << std::endl;
    std::cout << " Early termination after finding " << max_cor_early_term << " correspondences " << std::endl;
    N_3D = _N_3D;
    std::cout << " Query expansion includes the next " << N_3D << " features " << std::endl;
    filter_points = true;
    use_image_set_cover = true;
    consider_K_nearest_cams = _consider_K_nearest_cams;

    vw_handler.set_nb_trees(1);
    vw_handler.set_nb_visual_words(nb_clusters);
    vw_handler.set_branching(10);

    // store for every visual word at the finest level the id of its parents at levels 2 and 3
    parents_at_level_2 = new uint32_t[nb_clusters];
    parents_at_level_3 = new uint32_t[nb_clusters];

    for (uint32_t i = 0; i < nb_clusters; ++i)
    {
        parents_at_level_2[i] = parents_at_level_3[i] = nb_clusters;
    }

    vw_handler.set_method(std::string("flann"));
    vw_handler.set_flann_type(std::string("hkmeans"));

    if (!vw_handler.create_flann_search_index(cluster_file))
    {
        std::cout << " ERROR: Could not load the cluster centers from " << cluster_file << std::endl;
        return -1;
    }

    {
        // get the ids of the parents that we need
        int *parent_ids = new int[nb_clusters];
        for (uint32_t i = 0; i < nb_clusters; ++i)
            parent_ids[i] = -1;

        vw_handler.get_parents_at_level_L(2, parent_ids);

        // copy that to the real data structures
        uint32_t counter = 0;
        for (uint32_t i = 0; i < nb_clusters; ++i)
        {
            if (parent_ids[i] == -1)
            {
                std::cout << " uhoh, this should not happen!" << std::endl;
                ++counter;
            }
            else
            {
                parents_at_level_2[i] = (uint32_t)parent_ids[i];
                if (parents_at_level_2[i] < 0 || parents_at_level_2[i] >= 100)
                    std::cout << " WARNING: OUT OF RANGE: " << parents_at_level_2[i] << " from " << parent_ids[i] << std::endl;
            }
        }
        if (counter == nb_clusters)
        {
            std::cerr << " Some ERROR in getting the parents from level 2, stopping here " << std::endl;
            return -1;
        }

        for (uint32_t i = 0; i < nb_clusters; ++i)
            parent_ids[i] = -1;

        vw_handler.get_parents_at_level_L(3, parent_ids);

        // copy that
        counter = 0;
        for (uint32_t i = 0; i < nb_clusters; ++i)
        {
            if (parent_ids[i] == -1)
            {
                std::cout << " uhoh, this should not happen!" << std::endl;
                ++counter;
            }
            else
            {
                parents_at_level_3[i] = (uint32_t)parent_ids[i];
                if (parents_at_level_3[i] < 0 || parents_at_level_3[i] >= 1000)
                    std::cout << " WARNING: OUT OF RANGE: " << parents_at_level_3[i] << " from " << parent_ids[i] << std::endl;
            }
        }
        if (counter == nb_clusters)
        {
            std::cerr << " Some ERROR in getting the parents from level 3, stopping here " << std::endl;
            return -1;
        }
    }

    std::cout << "  done " << std::endl;

    ////
    // load the assignments of the 3D points to the visual words

    std::cout << "* Loading and parsing the assignments ... " << std::endl;

    vw_points_descriptors.resize(nb_clusters);

    nb_points_per_vw.resize(nb_clusters);
    std::fill(nb_points_per_vw.begin(), nb_points_per_vw.end(), 0);

    for (uint32_t i = 0; i < nb_clusters; ++i)
        vw_points_descriptors[i].clear();

    {
        std::ifstream ifs(vw_assignments.c_str(), std::ios::in | std::ios::binary);

        if (!ifs)
        {
            std::cerr << " ERROR: Cannot read the visual word assignments " << vw_assignments << std::endl;
            ;
            return 1;
        }

        uint32_t nb_clusts;
        ifs.read((char *)&nb_3D_points, sizeof(uint32_t));
        ifs.read((char *)&nb_clusts, sizeof(uint32_t));
        ifs.read((char *)&nb_non_empty_vw, sizeof(uint32_t));
        ifs.read((char *)&nb_descriptors, sizeof(uint32_t));
        if (nb_clusts != nb_clusters)
        {
            std::cerr << " WARNING: Number of clusters differs! " << nb_clusts << " " << nb_clusters << std::endl;
        }
        std::cout << "size of unsigned int on this platform: " << sizeof(uint32_t) << std::endl;
        std::cout << "Number of points: " << nb_3D_points << ".\nNumber of clusters: " << nb_clusts << ".\nNumber of non-empty clusters: " << nb_non_empty_vw << ".\nNumber of descriptors: " << nb_descriptors << std::endl;

        // read the 3D points and their visibility polygons
        points3D = new ANNcoord *[nb_3D_points];
        all_descriptors.resize(128 * nb_descriptors);

        // load the points
        std::cout << "loading points ... " << std::endl;
        colors_3D.resize(nb_3D_points);
        float *point_data = new float[3];
        unsigned char *color_data = new unsigned char[3];

        for (uint32_t i = 0; i < nb_3D_points; ++i)
        {
            points3D[i] = new ANNcoord[3];
            ifs.read((char *)point_data, 3 * sizeof(float));
            for (int j = 0; j < 3; ++j)
                points3D[i][j] = (ANNcoord)point_data[j];

            ifs.read((char *)color_data, 3 * sizeof(unsigned char));
            for (int j = 0; j < 3; ++j)
                colors_3D[i][j] = color_data[j];
        }
        delete[] point_data;
        delete[] color_data;

        desc_per_point.resize(nb_3D_points);
        vws_per_point.resize(nb_3D_points);
        for (uint32_t i = 0; i < nb_3D_points; ++i)
        {
            desc_per_point[i].clear();
            vws_per_point[i].clear();
        }
        std::cout << "loading descriptors ... " << std::endl;
        // load the descriptors
        int tmp_int;
        for (uint32_t i = 0; i < nb_descriptors; ++i)
        {
            for (uint32_t j = 0; j < 128; ++j)
                ifs.read((char *)&all_descriptors[128 * i + j], sizeof(unsigned char));
        }

        std::cout << "loading assignments of the pairs (point_id, descriptor_id) to the visual words ... " << std::endl;
        // now we load the assignments of the pairs (point_id, descriptor_id) to the visual words
        for (uint32_t i = 0; i < nb_non_empty_vw; ++i)
        {
            uint32_t id, nb_pairs;
            ifs.read((char *)&id, sizeof(uint32_t));
            ifs.read((char *)&nb_pairs, sizeof(uint32_t));
            vw_points_descriptors[id].resize(nb_pairs);
            nb_points_per_vw[id] = nb_pairs;
            for (uint32_t j = 0; j < nb_pairs; ++j)
            {
                ifs.read((char *)&vw_points_descriptors[id][j].first, sizeof(uint32_t));
                ifs.read((char *)&vw_points_descriptors[id][j].second, sizeof(uint32_t));

                // for every 3D point, remember the indices of its descriptors
                desc_per_point[vw_points_descriptors[id][j].first].push_back(vw_points_descriptors[id][j].second);
                vws_per_point[vw_points_descriptors[id][j].first].push_back(id);
            }
        }

        ifs.close();

        std::cout << "  done loading and parsing the assignments " << std::endl;
    }

    ////
    // load information about the 3D points:
    // the connected component it belongs to and the ids of the images it is observed in
    // we obtain all these information by parsing a Bundler file

    // for every 3D point, store the id of its connected component and the ids of the images that see the point
    connected_component_id_per_point.resize(nb_3D_points);
    std::fill(connected_component_id_per_point.begin(), connected_component_id_per_point.end(), 0);

    images_per_point.resize(nb_3D_points);

    for (uint32_t i = 0; i < nb_3D_points; ++i)
        images_per_point[i].clear();

    {
        // parse the reconstruction
        parse_bundler parser;
        if (!parser.parse_data(bundle_file.c_str(), 0))
        {
            std::cerr << " ERROR: Could not parse the bundler file " << bundle_file << std::endl;
            return -1;
        }
        uint32_t nb_cameras = parser.get_number_of_cameras();
        uint32_t nb_points_bundler = parser.get_number_of_points();
        std::vector<feature_3D_info> &feature_infos = parser.get_feature_infos();

        if (nb_points_bundler != nb_3D_points)
        {
            std::cerr << " ERROR: The number of points in the binary file ( " << nb_3D_points << " ) and in the reconstruction ( " << nb_points_bundler << " ) differ!" << std::endl;
            return -1;
        }

        // compute the connected components
        std::cout << "  Computing connected components " << std::endl;
        // for every camera, get the corresponding keypoint ids
        std::vector<std::vector<uint32_t>> cam_keys(nb_cameras);
        for (uint32_t i = 0; i < nb_cameras; ++i)
            cam_keys[i].clear();

        for (uint32_t i = 0; i < nb_points_bundler; ++i)
        {
            for (size_t j = 0; j < feature_infos[i].view_list.size(); ++j)
                cam_keys[feature_infos[i].view_list[j].camera].push_back(i);
        }

        std::vector<int> cam_ccs(nb_cameras, -1);
        std::vector<int> point_ccs(nb_points_bundler, -1);
        int nb_ccs = 0;

        std::queue<uint32_t> remaining_points;

        for (uint32_t i = 0; i < nb_points_bundler; ++i)
            remaining_points.push(i);

        while (!remaining_points.empty())
        {
            uint32_t cur_point = remaining_points.front();

            remaining_points.pop();

            if (point_ccs[cur_point] != -1)
                continue;

            std::queue<uint32_t> recursive_points;

            // create a new connected component
            point_ccs[cur_point] = nb_ccs;
            ++nb_ccs;

            for (size_t j = 0; j < feature_infos[cur_point].view_list.size(); ++j)
            {
                uint32_t cam_id = feature_infos[cur_point].view_list[j].camera;
                if (cam_ccs[cam_id] != -1 && cam_ccs[cam_id] != point_ccs[cur_point])
                    std::cout << " ERROR: ambigous ids for camera! " << std::endl;

                if (cam_ccs[cam_id] == -1)
                {
                    cam_ccs[cam_id] = point_ccs[cur_point];
                    for (size_t k = 0; k < cam_keys[cam_id].size(); ++k)
                    {
                        if (point_ccs[cam_keys[cam_id][k]] != -1 && point_ccs[cam_keys[cam_id][k]] != point_ccs[cur_point])
                            std::cout << " ERROR: ambigous ids for point " << std::endl;

                        if (point_ccs[cam_keys[cam_id][k]] == -1)
                            recursive_points.push(cam_keys[cam_id][k]);
                    }
                }
            }

            while (!recursive_points.empty())
            {
                uint32_t c = recursive_points.front();
                recursive_points.pop();

                if (point_ccs[c] != -1 && point_ccs[c] != point_ccs[cur_point])
                    std::cout << " ERROR: ambigous ids for point " << std::endl;

                if (point_ccs[c] == -1)
                {
                    // create a new connected component
                    point_ccs[c] = point_ccs[cur_point];

                    for (size_t j = 0; j < feature_infos[c].view_list.size(); ++j)
                    {
                        uint32_t cam_id = feature_infos[c].view_list[j].camera;
                        if (cam_ccs[cam_id] != -1 && cam_ccs[cam_id] != point_ccs[cur_point])
                            std::cout << " ERROR: ambigous ids for camera! " << std::endl;

                        if (cam_ccs[cam_id] == -1)
                        {
                            cam_ccs[cam_id] = point_ccs[cur_point];
                            for (size_t k = 0; k < cam_keys[cam_id].size(); ++k)
                            {
                                if (point_ccs[cam_keys[cam_id][k]] != -1 && point_ccs[cam_keys[cam_id][k]] != point_ccs[cur_point])
                                    std::cout << " ERROR: ambigous ids for point " << std::endl;

                                if (point_ccs[cam_keys[cam_id][k]] == -1)
                                    recursive_points.push(cam_keys[cam_id][k]);
                            }
                        }
                    }
                }
            }
        }
        std::cout << "   Found " << nb_ccs << " connected components " << std::endl;

        // check if all non-empty images and all points belong to a connected component
        for (uint32_t i = 0; i < nb_points_bundler; ++i)
        {
            if (point_ccs[i] == -1 || point_ccs[i] >= nb_ccs)
            {
                std::cout << " ERROR : Point " << i << " has connected component id " << point_ccs[i] << std::endl;
                return -1;
            }
        }

        for (uint32_t i = 0; i < nb_cameras; ++i)
        {
            if ((cam_ccs[i] == -1 || cam_ccs[i] >= nb_ccs) && (cam_keys[i].size() > 0))
            {
                std::cout << " ERROR : Non-empty camera " << i << " has connected component id " << cam_ccs[i] << std::endl;
                return -1;
            }

            // check if any of the points visible in this camera has a different conncected component id
            for (size_t k = 0; k < cam_keys[i].size(); ++k)
            {
                if (point_ccs[cam_keys[i][k]] != cam_ccs[i])
                {
                    std::cerr << " ERROR : Found point " << cam_keys[i][k] << " in camera " << i << " that belongs to component " << point_ccs[cam_keys[i][k]] << " while the camera belongs to component " << cam_ccs[i] << std::endl;
                    return -1;
                }
            }

            if (use_image_set_cover)
                std::sort(cam_keys[i].begin(), cam_keys[i].end());
            else
                cam_keys[i].clear();
        }

        ////
        // if we want to represent the set of images with a smaller set, we now compute it

        // recall for every image by which other images it is covered
        std::vector<std::set<uint32_t>> image_covered_by;
        image_covered_by.clear();
        std::vector<std::set<uint32_t>> images_covered_by_image;
        images_covered_by_image.clear();

        if (use_image_set_cover)
        {
            image_covered_by.resize(nb_cameras);

            images_covered_by_image.resize(nb_cameras);

            std::cout << "  Computing the set cover for all images, each image covers itself and (at most) the " << consider_K_nearest_cams << " images that have the largest number of 3D points in common with it " << std::endl;

            for (uint32_t i = 0; i < nb_cameras; ++i)
            {
                images_covered_by_image[i].clear();
                images_covered_by_image[i].insert(i);
            }

            ////
            // now add cameras that are close in 3D and have a similar viewing direction (angle between directions beneath 60°)
            std::vector<OpenMesh::Vec3f> camera_positions(nb_cameras);
            std::vector<OpenMesh::Vec3f> cam_viewing_dirs(nb_cameras);

            std::vector<bundler_camera> &bundle_cams = parser.get_cameras();

            for (uint32_t i = 0; i < nb_cameras; ++i)
            {
                camera_positions[i] = bundle_cams[i].get_cam_position_f();
                cam_viewing_dirs[i] = bundle_cams[i].get_cam_global_vec_f(OpenMesh::Vec3d(0.0, 0.0, -1.0));
            }

            for (uint32_t i = 0; i < nb_cameras; ++i)
            {
                std::vector<std::pair<uint32_t, float>> cameras_distances(nb_cameras);

                for (uint32_t j = 0; j < nb_cameras; ++j)
                {
                    cameras_distances[j].first = j;
                    cameras_distances[j].second = (camera_positions[i] - camera_positions[j]).length();
                }

                std::sort(cameras_distances.begin(), cameras_distances.end(), ACSLocalizer::cmp_second_entry_less<uint32_t, float>);

                // now pick the 10 nearest cameras looking in a similar direction out of the nearest 20
                // cameras from the same connected component
                uint32_t counter = 0;
                uint32_t found = 0;
                for (uint32_t j = 0; j < nb_cameras && counter < consider_K_nearest_cams /*&& found<5*/; ++j)
                {
                    // take care that the camera we are looking at is not camera i but is in the
                    // same connected component!
                    if (cameras_distances[j].first == i || cam_ccs[i] != cam_ccs[cameras_distances[j].first])
                        continue;

                    // remember we only want to look at the 20 nearest cameras and pick at most 10 from them
                    ++counter;

                    if ((cam_viewing_dirs[i] | cam_viewing_dirs[cameras_distances[j].first]) >= 0.5f)
                    {
                        images_covered_by_image[i].insert(cameras_distances[j].first);
                        ++found;
                    }
                }
            }

            camera_positions.clear();
            cam_viewing_dirs.clear();

            ////
            // now we compute the set cover

            // for every image, track how many new images it can cover
            std::vector<std::pair<uint32_t, uint32_t>> nb_new_images_covered(nb_cameras);

            for (uint32_t i = 0; i < nb_cameras; ++i)
            {
                image_covered_by[i].clear();
                nb_new_images_covered[i].first = i;
                nb_new_images_covered[i].second = images_covered_by_image[i].size();
            }

            // map image ids to a smaller range
            std::vector<int> new_image_ids(nb_cameras, -1);
            int size_set_cover = 0;

            while (!nb_new_images_covered.empty())
            {
                std::sort(nb_new_images_covered.begin(), nb_new_images_covered.end(), ACSLocalizer::cmp_second_entry_less<uint32_t, uint32_t>);
                if (nb_new_images_covered.back().second == 0)
                    break;

                uint32_t cam_id_ = nb_new_images_covered.back().first;

                // add new image to set cover
                new_image_ids[cam_id_] = size_set_cover;

                // markt its images as covered
                for (std::set<uint32_t>::const_iterator it = images_covered_by_image[cam_id_].begin(); it != images_covered_by_image[cam_id_].end(); ++it)
                    image_covered_by[*it].insert((uint32_t)size_set_cover);

                ++size_set_cover;

                // pop first element
                nb_new_images_covered.pop_back();

                // recompute the nb of new images each image can cover
                for (std::vector<std::pair<uint32_t, uint32_t>>::iterator it = nb_new_images_covered.begin(); it != nb_new_images_covered.end(); ++it)
                {
                    it->second = 0;
                    for (std::set<uint32_t>::const_iterator it2 = images_covered_by_image[it->first].begin(); it2 != images_covered_by_image[it->first].end(); ++it2)
                    {
                        if (image_covered_by[*it2].empty())
                            it->second += 1;
                    }
                }
            }

            nb_new_images_covered.clear();

            std::cout << "   Set cover contains " << size_set_cover << " cameras out of " << nb_cameras << std::endl;
        }
        else
        {
            image_covered_by.clear();
            images_covered_by_image.clear();
        }

        // copy the information
        for (uint32_t i = 0; i < nb_points_bundler; ++i)
            connected_component_id_per_point[i] = point_ccs[i];

        cam_ccs.clear();
        point_ccs.clear();

        std::cout << "  Reading the images per 3D point " << std::endl;
        for (uint32_t i = 0; i < nb_points_bundler; ++i)
        {
            images_per_point[i].clear();

            for (size_t j = 0; j < feature_infos[i].view_list.size(); ++j)
            {
                if (use_image_set_cover)
                {
                    uint32_t cam_id_ = feature_infos[i].view_list[j].camera;
                    for (std::set<uint32_t>::const_iterator it = image_covered_by[cam_id_].begin(); it != image_covered_by[cam_id_].end(); ++it)
                        images_per_point[i].insert(*it);
                }
                else
                    images_per_point[i].insert(feature_infos[i].view_list[j].camera);
            }

            if (images_per_point[i].size() == 0)
                std::cout << " WARNING: Point " << i << " is visible in no image!" << std::endl;
        }

        // clean up
        if (use_image_set_cover)
        {
            for (uint32_t i = 0; i < nb_cameras; ++i)
            {
                image_covered_by[i].clear();
                images_covered_by_image[i].clear();
            }
            image_covered_by.clear();
            images_covered_by_image.clear();
        }

        parser.clear();
        feature_infos.clear();
    }

    ////
    // create the kd-trees for the 3D points to enable search in 3D, one for each connected component

    // get the number of connected components
    nb_connected_components = *std::max_element(connected_component_id_per_point.begin(), connected_component_id_per_point.end());
    nb_connected_components += 1;

    std::cout << " * creating kd-trees for 3D points, one for each of the " << nb_connected_components << " connected components " << std::endl;

    nb_points_per_component.resize(nb_connected_components);
    std::fill(nb_points_per_component.begin(), nb_points_per_component.end(), 0);

    for (std::vector<uint32_t>::const_iterator it = connected_component_id_per_point.begin(); it != connected_component_id_per_point.end(); ++it)
        nb_points_per_component[*it] += 1;

    // for every connected component, get pointers to its 3D points
    //  double **points_per_component[nb_connected_components];

    points_per_component.resize(nb_connected_components);

    indices_per_component.clear();
    indices_per_component.resize(nb_connected_components);

    // store pointers to the appropriate points
    {
        std::vector<uint32_t> cc_point_counter(nb_connected_components, 0);

        for (uint32_t i = 0; i < nb_connected_components; ++i)
        {
            points_per_component[i] = new double *[nb_points_per_component[i]];
            indices_per_component[i].resize(nb_points_per_component[i]);
        }

        for (uint32_t i = 0; i < nb_3D_points; ++i)
        {
            uint32_t cc_id = connected_component_id_per_point[i];
            points_per_component[cc_id][cc_point_counter[cc_id]] = (double *)points3D[i];
            indices_per_component[cc_id][cc_point_counter[cc_id]] = i;
            cc_point_counter[cc_id] += 1;
        }
    }

    // create the trees
    // ANNkd_tree *kd_trees[nb_connected_components];

    kd_trees.resize(nb_connected_components);

    for (uint32_t i = 0; i < nb_connected_components; ++i)
    {
        int nb_points_3D_int = (int)nb_points_per_component[i];
        kd_trees[i] = new ANNkd_tree(points_per_component[i], nb_points_3D_int, 3, 1);
    }

    int nb_points_3D_int = (int)nb_3D_points;
    kd_tree = new ANNkd_tree(points3D, nb_points_3D_int, 3);

    // and the search structures
    annMaxPtsVisit(0);

    indices = new ANNidx[N_3D];
    distances = new ANNdist[N_3D];

    std::cout << "  done " << std::endl;
};
}