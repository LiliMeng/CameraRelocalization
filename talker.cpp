#include "ros/ros.h"
#include "std_msgs/String.h"


#include <sstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "cameraParameters.h"
#include "pointDefinition.h"

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
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

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>    // OpenCV
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

using namespace sensor_msgs;
using namespace message_filters;

using namespace std;
using namespace cv;

static const std::string OPENCV_WINDOW = "Image window";

//#define EXACT
#define APPROXIMATE


#ifdef EXACT
#include <message_filters/sync_policies/exact_time.h>
#endif
#ifdef APPROXIMATE
#include <message_filters/sync_policies/approximate_time.h>
#endif



using namespace std;
//using namespace sensor_msgs;
using namespace message_filters;


// Contador para la numeración de los archivos.
// Counter for filenames.
unsigned int cnt = 1;

bool camera_relocalization()
{
   const char * model_file = "/home/ial/segbot_nav_ws/devel/lib/camera_relocalization/model/bt_rgbd_RF.txt";
   const char * rgb_image_file = "/home/ial/segbot_nav_ws/devel/lib/camera_relocalization/train_data/rgb_image_list.txt";
   const char * depth_image_file = "/home/ial/segbot_nav_ws/devel/lib/camera_relocalization/train_data/depth_image_list.txt";
   const char * camera_to_wld_pose_file = "/home/ial/segbot_nav_ws/devel/lib/camera_relocalization/train_data/camera_pose_list.txt";
   const int num_random_sample = 5000;
   const int max_check = 8;
   const char * dataset_param_filename = "/home/ial/segbot_nav_ws/devel/lib/camera_relocalization/4scenes_param.txt";
    
  
   const double inlierFeatDist = 0.3;
   const double inlierThreshold = 0.1;
   const double angleThreshold    = 5;
   const double distanceThreshold = 0.05;

   assert(num_random_sample > 100);
    
    vector<string> rgb_files   = Ms7ScenesUtil::read_file_names(rgb_image_file);
    vector<string> depth_files = Ms7ScenesUtil::read_file_names(depth_image_file);
    vector<string> pose_files  = Ms7ScenesUtil::read_file_names(camera_to_wld_pose_file);
    
    assert(rgb_files.size() == depth_files.size());
    assert(rgb_files.size() == pose_files.size());
    
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
    
    vector<double> angle_errors;
    vector<double> translation_errors;
    
    vector<cv::Mat> estimated_poses;

    // read images, and predict one by one
    for (int k = 0; k<rgb_files.size(); k++)
    {
        clock_t begin1=clock();
        
        const char *cur_rgb_img_file     = rgb_files[k].c_str();
        const char *cur_depth_img_file   = depth_files[k].c_str();
        const char *cur_pose_file        = pose_files[k].c_str();
        
        cv::Mat rgb_img;
        CvxIO::imread_rgb_8u(cur_rgb_img_file, rgb_img);
        vector<FeatureType>     features;
        vector<Eigen::VectorXf> labels;
        BTRNDUtil::randomSampleFromRgbdImages(cur_rgb_img_file, cur_depth_img_file, cur_pose_file,
                                              num_random_sample, k, dataset_param,
                                              is_use_depth, false,
                                              features, labels);
        BTRNDUtil::extractWHFeatureFromRgbImages(cur_rgb_img_file, features, wh_kernel_size, false);
        assert(features.size() == labels.size());
        
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
                all_labels.push_back(labels[j]);
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
    
        vector<cv::Point3d> wld_pts_gt;
        for(int m=0; m<all_labels.size(); m++)
        {
            double x_gt_world = all_labels[m](0);
            double y_gt_world = all_labels[m](1);
            double z_gt_world = all_labels[m](2);
            wld_pts_gt.push_back(cv::Point3d(x_gt_world, y_gt_world, z_gt_world));
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
    
        string depth_img_file = depth_files[k];
        string camera_pose_file = pose_files[k];
        cv::Mat depth_img;
        CvxIO::imread_depth_16bit_to_64f(depth_img_file.c_str(), depth_img);
        
        cv::Mat camera_to_world_pose = Ms7ScenesUtil::read_pose_7_scenes(camera_pose_file.c_str());
        
        cv::Mat mask;
        cv::Mat camera_coordinate_position;
        cv::Mat wld_coord = Ms7ScenesUtil::cameraDepthToWorldCoordinate(depth_img,
                                                                        camera_to_world_pose,
                                                                        calibration_matrix,
                                                                        depth_factor,
                                                                        min_depth,
                                                                        max_depth,
                                                                        camera_coordinate_position,
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
        if (valid_camera_pts.size() < 20) {
            angle_errors.push_back(180.0);
            translation_errors.push_back(10.0);
            estimated_poses.push_back(estimated_camera_pose);
            continue;
        }
        
        // estimate camera pose using Kabsch
        PreemptiveRANSAC3DParameter param;
        param.dis_threshold_ = inlierThreshold;
        bool isEstimated = CvxPoseEstimation::preemptiveRANSAC3DOneToMany(valid_camera_pts, valid_wld_pts_candidate, param, estimated_camera_pose);
        if (isEstimated) {
            double angle_dis = 0.0;
            double location_dis = 0.0;
            cv::Mat gt_pose = Ms7ScenesUtil::read_pose_7_scenes(camera_pose_file.c_str());
            CvxPoseEstimation::poseDistance(gt_pose, estimated_camera_pose, angle_dis, location_dis);
            angle_errors.push_back(angle_dis);
            translation_errors.push_back(location_dis);
            printf("angle distance, location distance are %lf %lf\n", angle_dis, location_dis);
        }
        else
        {
            angle_errors.push_back(180.0);
            translation_errors.push_back(10.0);
        }
        
        cout<<"estimated_camera_pose"<<estimated_camera_pose<<endl;
        clock_t end2 = clock();
        double feature_extraction_time = double(begin2-begin1)/(double)CLOCKS_PER_SEC;
        double forest_prediction_time = double(begin3-begin2)/(double)CLOCKS_PER_SEC;
        double test_estimate_time = double(end2 - begin2)/(double)CLOCKS_PER_SEC;
        cout.precision(5);
        cout<<"feature extraction time "<<feature_extraction_time<<endl;
        cout<<"forest prediction time "<<forest_prediction_time<<endl;
        cout<<"camera relocalization time "<<test_estimate_time<<endl;

        estimated_poses.push_back(estimated_camera_pose);
        
    }
    return 0;
}


class ImageConverter
{

  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber rgb_image_sub_;
  image_transport::Subscriber depth_image_sub_;
  image_transport::Publisher image_pub_;
  cv_bridge::CvImagePtr rgbImgPtr;
  cv_bridge::CvImagePtr depthImgPtr;
 
public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscribe to input video feed and publish output video feed
    rgb_image_sub_ = it_.subscribe("/camera/rgb/image_rect_color", 1, &ImageConverter::rgb_imageCb, this);
    depth_image_sub_ = it_.subscribe("/camera/depth/image", 1, &ImageConverter::depth_imageCb, this);

    cv::namedWindow("rgb_image_window", WINDOW_AUTOSIZE);
    cv::namedWindow("depth_image_window", WINDOW_AUTOSIZE);
  
   
  }

  ~ImageConverter()
  {
    cv::destroyWindow("rgb_image_window");
    cv::destroyWindow("depth_image_window");
  }

 


  void rgb_imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
   
    try
    {
      rgbImgPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

      // Update GUI Window
    cv::imshow("rgb_image_window", rgbImgPtr->image);
    cv::waitKey(3);
   
  
  }

   void depth_imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    
    try
    {
       depthImgPtr = cv_bridge::toCvCopy(msg , sensor_msgs::image_encodings::TYPE_32FC1);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // Update GUI Window
    cv::imshow("depth_image_window", depthImgPtr->image);
    cv::waitKey(3);
    
    // Output modified video stream
    //image_pub_.publish(cv_ptr->toImageMsg());

  //  camera_relocalization();
  }

};

 void rgbd_image_callback(const sensor_msgs::ImageConstPtr& rgb_msg, const sensor_msgs::ImageConstPtr& depth_msg)
 {
    cv_bridge::CvImagePtr rgbImgPtr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
    cv_bridge::CvImagePtr depthImgPtr = cv_bridge::toCvCopy(depth_msg , sensor_msgs::image_encodings::TYPE_32FC1);
    
    //Normalize the pixel value
    cv::normalize(depthImgPtr->image, depthImgPtr->image, 1, 0, cv::NORM_MINMAX);
   
   /* cv::namedWindow("rgb_image_window", WINDOW_AUTOSIZE);
    cv::namedWindow("depth_image_window", WINDOW_AUTOSIZE);
    cv::imshow("rgb_image_window", rgbImgPtr->image);
     cv::waitKey(3);
    cv::imshow("depth_image_window", depthImgPtr->image);
    cv::waitKey(3);
    */
  
    const char * model_file = "/home/ial/segbot_nav_ws/devel/lib/camera_relocalization/model/bt_rgbd_RF.txt";
   // const char*  rgb_image_file = "/home/ial/segbot_nav_ws/devel/lib/camera_relocalization/test_data/rgb_image_list.txt";
   // const char*  depth_image_file = "/home/ial/segbot_nav_ws/devel/lib/camera_relocalization/test_data/depth_image_list.txt";
   // const char*  camera_to_wld_pose_file = "/home/ial/segbot_nav_ws/devel/lib/camera_relocalization/test_data/camera_pose_list.txt";
    const int num_random_sample = 5000;
    const int max_check = 8;
    const char* dataset_param_filename = "4scenes_param.txt";
    
    const double inlierFeatDist = 0.3;
    const double inlierThreshold = 0.1;
    const double angleThreshold    = 5;
    const double distanceThreshold = 0.05;
   
   
    cv::Mat rgb_img= rgbImgPtr->image;
    cv::Mat depth_img= depthImgPtr->image; 
    
    assert(depth_img.type() == CV_32FC1);
    depth_img.convertTo(depth_img, CV_64F);
  
    assert(num_random_sample > 100);
    
    // read model
    BTRNDRegressor model;
    bool is_read = model.load(model_file);
    if (!is_read) {
        printf("Error: can not read from file %s\n", model_file);

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
    
    
     BTRNDUtil::extractWHFeatureFromRgbImages(rgb_img, features, wh_kernel_size, false);
    
        
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
   
    
 } 

int main(int argc, char **argv)
{
  
  ros::init(argc, argv, "talker");
  
  ros::NodeHandle nh;
  
    message_filters::Subscriber<Image> rgb_image_sub(nh, "/camera/rgb/image_rect_color", 1);
    message_filters::Subscriber<Image> depth_image_sub(nh, "/camera/depth/image", 1);
  
    typedef sync_policies::ApproximateTime<Image, Image> MySyncPolicy;
   // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), rgb_image_sub, depth_image_sub);
    sync.registerCallback(boost::bind(&rgbd_image_callback, _1, _2));

  //ImageConverter IC;
  
  
  ros::Publisher chatter_pub = nh.advertise<std_msgs::String>("chatter", 1000);
 
 
  ros::Rate loop_rate(1);

  int count = 0;
  
 //camera_relocalization();
    
    

  //while (ros::ok())
  //{
    /**
     * This is a message object. You stuff it with data, and then publish it.
     */
   // std_msgs::String msg;

  //  std::stringstream ss;
   // ss << "hello world " << count;
   // msg.data = ss.str();

   // ROS_INFO("%s", msg.data.c_str());

   // chatter_pub.publish(msg);

  //  ros::spinOnce();

  //  loop_rate.sleep();
     
   // ++count;
 // }

  ros::spin();                                    
  return 0;
}
