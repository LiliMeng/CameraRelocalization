//
//  main.cpp
//  LoopClosure
//
//  Created by Lili on 2017-02-16.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include <iostream>
#include "cvxImage_310.hpp"
#include <string>
#include "cvxIO.hpp"
#include <unordered_map>
#include "ms7ScenesUtil.hpp"
#include "bt_rnd_regressor_builder.h"
#include "bt_rnd_regressor.h"
#include "cvxImage_310.hpp"
#include "cvxIO.hpp"
#include "cvxPoseEstimation.hpp"
#include "ms7ScenesUtil.hpp"
#include "dataset_param.h"


using std::string;

#if 1

cv::Mat validCameraDepth(const cv::Mat & camera_depth_img,
                         const double depth_factor,
                          const double min_depth,
                          const double max_depth,
                          cv::Mat & mask)
{
    assert(camera_depth_img.type() == CV_64FC1);
    assert(min_depth < max_depth);
    assert(min_depth >= 0.0);
    
    const int width  = camera_depth_img.cols;
    const int height = camera_depth_img.rows;
  
    cv::Mat loc_img = cv::Mat::zeros(3, 1, CV_64F);
    mask = cv::Mat::ones(height, width, CV_8UC1);
        for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double camera_depth = camera_depth_img.at<double>(r, c)/depth_factor; // to meter
            if (camera_depth < min_depth || camera_depth > max_depth ) {
                // invalid depth
                //printf("invalid depth %lf\n", camera_depth);
                mask.at<unsigned char>(r, c) = 0;
                continue;
            }
            loc_img.at<double>(0, 0) = c;
            loc_img.at<double>(1, 0) = r;
            loc_img.at<double>(2, 0) = 1.0;
        }
    }
    return loc_img;
    
}

int main(int argc, const char * argv[])
{
  
    const char * model_file = "model/bt_rgbd_RF.txt";
    const char*  rgb_image_file = "test_files/rgb_image_list.txt";
    const char*  depth_image_file = "test_files/depth_image_list.txt";
    const char*  camera_to_wld_pose_file = "test_files/camera_pose_list.txt";
    const int num_random_sample = 5000;
    const int max_check = 8;
    const char* dataset_param_filename = "4scenes_param.txt";
    
    const double inlierFeatDist = 0.3;
    const double inlierThreshold = 0.1;
    const double angleThreshold    = 5;
    const double distanceThreshold = 0.05;
   
    const char* cur_rgb_img_file = "/Users/jimmy/Desktop/images/4_scenes/apt1/living/rgb/frame-000001.color.png";
    const char* cur_depth_img_file = "/Users/jimmy/Desktop/images/4_scenes/apt1/living/data/frame-000001.depth.png";
    cv::Mat rgb_img, depth_img;
    
    CvxIO::imread_rgb_8u(cur_rgb_img_file, rgb_img);
    CvxIO::imread_depth_16bit_to_64f(cur_depth_img_file, depth_img);
   
  
    assert(num_random_sample > 100);
    
    // read model
    BTRNDRegressor model;
    bool is_read = model.load(model_file);
    if (!is_read) {
        printf("Error: can not read from file %s\n", model_file);
        return -1;
    }
    
    const BTRNDTreeParameter & tree_param = model.getTreeParameter();
    const DatasetParameter  & dataset_param = model.getDatasetParameter();
    const bool use_depth = tree_param.is_use_depth_;
    if (use_depth) {
        printf("use depth in the feature.\n");
    }
    else {
        printf("not use depth in the feature.\n");
    }
    
    dataset_param.printSelf();
    tree_param.printSelf();
    
    cv::Mat camera_matrix = dataset_param.camera_matrix();
    const int wh_kernel_size = tree_param.wh_kernel_size_;
    const bool is_use_depth = tree_param.is_use_depth_;
    
    
    cv::Mat calibration_matrix = dataset_param.camera_matrix();
    const double depth_factor = dataset_param.depth_factor_;
    const double min_depth = dataset_param.min_depth_;
    const double max_depth = dataset_param.max_depth_;
    
    using FeatureType = SCRFRandomFeature;
    
   
    
    vector<FeatureType>   features;
    
        
    clock_t begin1=clock();
    BTRNDUtil::randomSampleFromRgbdImages(rgb_img,
                                          depth_img,
                                          num_random_sample,
                                          dataset_param,
                                          is_use_depth,
                                          false,
                                          features);
    
    
     BTRNDUtil::extractWHFeatureFromRgbImages(cur_rgb_img_file, features, wh_kernel_size, false);
    
        
    // predict from the model
    vector<vector<Eigen::VectorXf> > all_predictions;
    vector<vector<float> > all_distances;
    vector<Eigen::VectorXf> all_labels;    // labels
    vector<Eigen::Vector2f> all_locations; // 2d location
        
    clock_t begin2 = clock();
        
        
    for(int j = 0; j<features.size(); j++)
    {
        vector<Eigen::VectorXf> preds;
        vector<float> dists;
        bool is_predict = model.predict(features[j], rgb_img, max_check, preds, dists);
            
        if(is_predict)
        {
            all_predictions.push_back(preds);
            all_distances.push_back(dists);
            all_locations.push_back(features[j].p2d_);
        }
    }
        
    clock_t begin3 = clock();
        
    vector<cv::Point2d> img_pts;
    for(int m=0; m<all_locations.size(); m++)
    {
        double x_img= all_locations[m](0);
        double y_img= all_locations[m](1);
        img_pts.push_back(cv::Point2d(x_img,y_img));
    }
    
    vector<vector<cv::Point3d> > wld_pts_pred_candidate;
    for(int m=0; m<all_predictions.size(); m++)
    {
        vector<cv::Point3d> tmp_wld_pred;
        for(int n=0; n<all_predictions[m].size(); n++)
        {
            double x_pred_world = all_predictions[m][n](0);
            double y_pred_world = all_predictions[m][n](1);
            double z_pred_world = all_predictions[m][n](2);
            tmp_wld_pred.push_back(cv::Point3d(x_pred_world, y_pred_world, z_pred_world));
        }
        wld_pts_pred_candidate.push_back(tmp_wld_pred);
    }
        
    clock_t begin4 = clock();
    
    cv::Mat mask;
    
    cv::Mat camera_coordinate_position = Ms7ScenesUtil::camera_depth_to_camera_coordinate(depth_img,
                                                                                          min_depth,
                                                                                          max_depth,
                                                                                          mask);
    
    // 2D location to 3D camera coordiante location*
    vector<vector<cv::Point3d> > valid_wld_pts_candidate;
    vector<cv::Point3d> valid_camera_pts;
    for(int i = 0; i<img_pts.size(); i++) {
        int x = img_pts[i].x;
        int y = img_pts[i].y;
        if(mask.at<unsigned char>(y, x) != 0) {
            cv::Point3d p = cv::Point3d(camera_coordinate_position.at<cv::Vec3d>(y, x));
            valid_camera_pts.push_back(p);
            valid_wld_pts_candidate.push_back(wld_pts_pred_candidate[i]);
        }
    }
    
        cv::Mat estimated_camera_pose = cv::Mat::eye(4, 4, CV_64F);
    
        // estimate camera pose using Kabsch
        PreemptiveRANSAC3DParameter param;
        param.dis_threshold_ = inlierThreshold;
        bool isEstimated = CvxPoseEstimation::preemptiveRANSAC3DOneToMany(valid_camera_pts, valid_wld_pts_candidate, param, estimated_camera_pose);
    
        cout<<"estimated_camera_pose"<<estimated_camera_pose<<endl;
        clock_t end2 = clock();
        double feature_extraction_time = double(begin2-begin1)/(double)CLOCKS_PER_SEC;
        double forest_prediction_time = double(begin3-begin2)/(double)CLOCKS_PER_SEC;
        double format_convert_time=double(begin4 - begin3)/(double)CLOCKS_PER_SEC;
        double test_estimate_time = double(end2 - begin2)/(double)CLOCKS_PER_SEC;
        cout.precision(5);
        cout<<"feature extraction time "<<feature_extraction_time<<endl;
        cout<<"forest prediction time "<<forest_prediction_time<<endl;
        cout<<"format conversion time "<<format_convert_time<<endl;
        cout<<"camera relocalization time "<<test_estimate_time<<endl;
    
  
    return 0;
}

#endif
