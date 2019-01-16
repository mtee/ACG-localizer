/*===========================================================================*\
 *                                                                           *
 *                            ACG Localizer                                  *
 *      Copyright (C) 2012 by Computer Graphics Group, RWTH Aachen           *
 *                           www.rwth-graphics.de                            *
 *                                                                           *
 *---------------------------------------------------------------------------*
 *  This file is part of ACG Localizer                                       *
 *                                                                           *
 *  ACG Localizer is free software: you can redistribute it and/or modify    *
 *  it under the terms of the GNU General Public License as published by     *
 *  the Free Software Foundation, either version 3 of the License, or        *
 *  (at your option) any later version.                                      *
 *                                                                           *
 *  ACG Localizer is distributed in the hope that it will be useful,         *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of           *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            *
 *  GNU General Public License for more details.                             *
 *                                                                           *
 *  You should have received a copy of the GNU General Public License        *
 *  along with ACG Localizer.  If not, see <http://www.gnu.org/licenses/>.   *
 *                                                                           *
\*===========================================================================*/
#define __STDC_LIMIT_MACROS

// C++ includes
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>

// 3D viewer
//#include "pointcloudmapping.h"

#include "acs_localizer.h"


int main(int argc, char **argv)
{

    if (argc != 5)
    {
        std::cout << "____________________________________________________________________________________________________________________________" << std::endl;
        std::cout << " -                                                                                                                        - " << std::endl;
        std::cout << " -        Localization method. Implementation of the localization framework proposed in the ECCV 2011 paper               - " << std::endl;
        std::cout << " -          T. Sattler, B. Leibe, L. Kobbelt. Improving Image-Based Localization by Active Correspondence Search.         - " << std::endl;
        std::cout << " -                               2012 by Torsten Sattler (tsattler@cs.rwth-aachen.de)                                     - " << std::endl;
        std::cout << " -                                                                                                                        - " << std::endl;
        std::cout << " - usage: acg_localizer_active_search list bundle_file nb_cluster clusters descriptors                                    - " << std::endl;
        std::cout << " - Parameters:                                                                                                            - " << std::endl;
        std::cout << " -  list                                                                                                                  - " << std::endl;
        std::cout << " -     List containing the filenames of all the .key files that should be used as query. It is assumed that the           - " << std::endl;
        std::cout << " -     corresponding images have the same filename except of ending in .jpg.                                              - " << std::endl;
        std::cout << " -                                                                                                                        - " << std::endl;
        std::cout << " -  bundle_file                                                                                                           - " << std::endl;
        std::cout << " -     The bundle.out file generated by Bundler.                                                                          - " << std::endl;
        std::cout << " -                                                                                                                        - " << std::endl;
        std::cout << " -  clusters                                                                                                              - " << std::endl;
        std::cout << " -     The cluster centers (visual words), stored in a textfile consisting of nb_clusters * 128 floating point values.    - " << std::endl;
        std::cout << " -                                                                                                                        - " << std::endl;
        std::cout << " -  descriptors                                                                                                           - " << std::endl;
        std::cout << " -     The assignments assigning descriptors (and 3D points) to visual words. It is assumed that all descriptors          - " << std::endl;
        std::cout << " -     are stored as unsigned char values and that every visual word contains at most one descriptor for every 3D point,  - " << std::endl;
        std::cout << " -     i.e., the integer mean strategy is used to represent 3D points.                                                    - " << std::endl;
        return 1;
    }

    // rocketcluster
    std::string keylist(argv[1]);
    std::string bundle_file(argv[2]);
    std::string cluster_file(argv[3]);
    std::string vw_assignments(argv[4]);
    pca::ACSLocalizer loc;
    // keylist = list.txt 
    // bundle_file = '/home/mikhail/Documents/RTAB-Map/bundler/voest_4300/cameras.out' 
    // cluster_file = /home/mikhail/workspace/ACG-localizer/markt_paris_gpu_sift_100k.cluster 
    // vw_assignments = '/home/mikhail/Documents/RTAB-Map/bundler/voest_4300/bundle.desc_assignments.integer_mean.voctree.clusters.100k.bin'
    loc.init(keylist, bundle_file, 100000, cluster_file, vw_assignments, 0, 500, 10);

//    PointCloudMapping pcm(100);
 //   pcm.AddPointCloud(points3D, colors_3D, 1.0);
   // pcm.AddPointCloud("/home/mikhail/Documents/RTAB-Map/bundler/voest_4300/cloud.ply");
    ////
    // now load the filenames of the query images
    std::vector<std::string> key_filenames;
    key_filenames.clear();
    {
        std::ifstream ifs(keylist.c_str(), std::ios::in);
        std::string tmp_string;

        while (!ifs.eof())
        {
            tmp_string = "";
            ifs >> tmp_string;
            if (!tmp_string.empty())
            {
                key_filenames.push_back(tmp_string);
            }
        }
        ifs.close();
        std::cout << " done loading " << key_filenames.size() << " keyfile names " << std::endl;
    }
    uint32_t nb_keyfiles = key_filenames.size();
    cv::Mat camMatrix_raw = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat camMatrix_undistorted = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_64F);

    // tango at 960p
    //  camMatrix.at<double>(0, 0) = 867.601196289;
    //  camMatrix.at<double>(1, 1) = 867.601196289;
    //  camMatrix.at<double>(0, 2) = 482.6857910;
    //  camMatrix.at<double>(1, 2) = 268.82540;
    //  camMatrix.at<double>(2, 2) = 1;

    // pupil at 720p
    // camMatrix_raw.at<double>(0, 0) = 910.25960;
    // camMatrix_raw.at<double>(1, 1) = 910.22312;
    // camMatrix_raw.at<double>(0, 2) = 608.785;
    // camMatrix_raw.at<double>(1, 2) = 403.538;
    // camMatrix_raw.at<double>(2, 2) = 1;

    // camMatrix_undistorted.at<double>(0, 0) = 910.25960;
    // camMatrix_undistorted.at<double>(1, 1) = 910.22312;
    // camMatrix_undistorted.at<double>(0, 2) = 1280 / 2;
    // camMatrix_undistorted.at<double>(1, 2) = 720 / 2;
    // camMatrix_undistorted.at<double>(2, 2) = 1;

    // distCoeffs.at<double>(0, 0) = -0.63037088;
    // distCoeffs.at<double>(0, 1) = 0.17767048;
    // distCoeffs.at<double>(0, 2) = -0.00489945;
    // distCoeffs.at<double>(0, 3) = -0.00192122;
    // distCoeffs.at<double>(0, 4) = 0.1757496;

    // distCoeffs.at<double>(0, 0) = -0.63037088;
    // distCoeffs.at<double>(0, 1) =  0.17767048;
    // distCoeffs.at<double>(0, 2) = -0.00489945;
    // distCoeffs.at<double>(0, 3) = -0.00192122;
    // distCoeffs.at<double>(0, 4) =  0.1757496;

    // galaxy s8 at 720p
    camMatrix_raw.at<double>(0, 0) = 1200 / 2;
    camMatrix_raw.at<double>(1, 1) = 1200 / 2;
    camMatrix_raw.at<double>(0, 2) = 1480 / 4;
    camMatrix_raw.at<double>(1, 2) = 720 / 4;
    camMatrix_raw.at<double>(2, 2) = 1;

    camMatrix_undistorted = camMatrix_raw;

   // std::ofstream ofs(outfile.c_str(), std::ios::out);

    // if (!ofs.is_open())
    // {
    //     std::cerr << " Could not write results to " << outfile << std::endl;
    //     return 1;
    // }

    uint32_t registered = 0;


    for (uint32_t i = 0; i < nb_keyfiles; ++i)
    {
        std::cout << std::endl
                  << " --------- " << i + 1 << " / " << nb_keyfiles << " --------- " << std::endl;

        //   SIFT_loader key_loader;
        std::cout << key_filenames[i] << std::endl;
        std::string jpg_filename(key_filenames[i]);
        //    key_loader.load_features( key_filenames[i].c_str(), LOWE );

        //    std::vector< unsigned char* >& descriptors = key_loader.get_descriptors();
        //    std::vector< SIFT_keypoint >& keypoints = key_loader.get_keypoints();
        std::cout << "loading query image: " << jpg_filename << std::endl;
        cv::Mat img_rgb_q_raw = cv::imread(jpg_filename, CV_LOAD_IMAGE_ANYCOLOR);
        cv::Mat img_q_rgb;
        cv::undistort(img_rgb_q_raw, img_q_rgb, camMatrix_raw, distCoeffs);

        cv::Mat img_gray_q;
        cv::cvtColor(img_q_rgb, img_gray_q, cv::COLOR_BGR2GRAY);
        cv::Mat inliers;
        std::vector<float> c2D, c3D;
        cv::Mat mDescriptors_q;
        std::set<size_t> unique_vw;
        cv::Mat trans;
        trans = loc.processImage(img_gray_q, camMatrix_undistorted, inliers, c2D, c3D, mDescriptors_q, unique_vw);
        std::cout << "Pose: " << std::endl;
        std::cout << trans << std::endl;
       // pcm.AddOrUpdateFrustum("2", trans, 1, 1, 0, 0, 2);
        uint32_t nb_corr = c2D.size() / 2;
        if (img_q_rgb.data)
        {
            double textSize = 1.0;
            int font = cv::FONT_HERSHEY_PLAIN;
            std::ostringstream str;
            str << "QFs: " << mDescriptors_q.rows;
            cv::putText(img_q_rgb,
                        str.str(),
                        cv::Point(5, 12), // Coordinates
                        font,             // Font
                        textSize,         // Scale. 2.0 = 2x bigger
                        cv::Scalar(0, 255, 0));
            str.str("");
            str.clear();
            str << "VWs: " << unique_vw.size();
            cv::putText(img_q_rgb,
                        str.str(),
                        cv::Point(5, 25), // Coordinates
                        font,             // Font
                        textSize,         // Scale. 2.0 = 2x bigger
                        cv::Scalar(0, 255, 0));
            str.str("");
            str.clear();

            str << "Corrs: " << nb_corr;
            cv::putText(img_q_rgb,
                        str.str(),
                        cv::Point(5, 37), // Coordinates
                        font,             // Font
                        textSize,         // Scale. 2.0 = 2x bigger
                        cv::Scalar(0, 255, 0));
            str.str("");
            str.clear();
            str << "Inliers: " << inliers.size();
            cv::putText(img_q_rgb,
                        str.str(),
                        cv::Point(5, 49), // Coordinates
                        font,             // Font
                        textSize,         // Scale. 2.0 = 2x bigger
                        cv::Scalar(0, 255, 0));
            cv::RNG rng(12345);
            for (int i = 0; i < c2D.size(); i += 2)
            {
                cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
                cv::circle(img_q_rgb, cv::Point(c2D[i] + (img_q_rgb.cols - 1.0) / 2.0f, -c2D[i + 1] + (img_q_rgb.rows - 1.0) / 2.0f), 4.0, color);
            }
            cv::imshow("Query Image", img_q_rgb);
            cv::waitKey(5);
        }
    }
  //  ofs.close();
    return 0;
}
