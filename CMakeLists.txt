cmake_minimum_required(VERSION 2.8.3)
project(camera_relocalization)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  message_filters
  cv_bridge
  pcl_ros
  image_transport
)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

catkin_package(
  CATKIN_DEPENDS
  roscpp
  std_msgs
  sensor_msgs
  message_filters
  cv_bridge
  pcl_ros
  image_transport
)



find_package(Boost COMPONENTS system filesystem REQUIRED)



# link Eigen
set(EIGEN_DIR /usr/local/include/Eigen)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

# link Flann
set(FLANN_DIR /home/ial/workspace/flann-1.8.4-src)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN_DIR}
  "${PROJECT_SOURCE_DIR}/include"
  ${FLANN_DIR}/src/cpp
  ${Boost_INCLUDE_DIR}
  /usr/local/include # Location when using 'make system_install'
  /usr/include       # More usual location (e.g. when installing using a package)
)

## Specify additional locations for library files
link_directories(
  /usr/local/lib # Location when using 'make system_install'
  /usr/lib       # More usual location (e.g. when installing using a package)
  ${FLANN_DIR}/build/lib
  ${Boost_LIBRARY_DIR}
)


# add source files
include_directories (./src ./src/DTR ./src/DTC ./src/BTDTR ./src/RF_BT_RND ./src/RND ./src/opencv_util ./src/Walsh_Hadamard ./src/opencv_util/Kabsch)


# .cpp in src
set(SOURCE_CODE1 ./src/DTUtil_IO.cpp ./src/ParameterParser.cpp ./src/yael_io.cpp)

# .cpp in opencv_util
set(SOURCE_CODE_OPENCV_UTIL ./src/opencv_util/cvxWalshHadamard.cpp ./src/opencv_util/ms7ScenesUtil.cpp ./src/opencv_util/cvxUtil.cpp ./src/opencv_util/cvxIO.cpp ./src/opencv_util/cvxPoseEstimation.cpp ./src/opencv_util/cvxCalib3d.cpp)

# .cpp in Kabsch
set(SOURCE_CODE_KABSCH ./src/opencv_util/Kabsch/Kabsch.cpp)

# DTR decision tree for regression
set(SOURCE_CODE_DTR ./src/DTR/DTRNode.cpp ./src/DTR/DTRTree.cpp ./src/DTR/DTRUtil.cpp ./src/DTR/DTRegressor.cpp ./src/DTR/DTRegressorBuilder.cpp)

# DTC decision tree for classification
set(SOURCE_CODE_DTC ./src/DTC/DTCNode.cpp ./src/DTC/DTCTree.cpp ./src/DTC/DTCUtil.cpp ./src/DTC/DTClassifier.cpp ./src/DTC/DTClassifierBuilder.cpp)

# BTDTR
set(SOURCE_CODE_BTDTR ./src/BTDTR/BTDTRUtil.cpp)

# Walsh hadamard feature
set(SOURCE_CODE_WH ./src/Walsh_Hadamard/defs.c ./src/Walsh_Hadamard/image.c ./src/Walsh_Hadamard/matrix.c ./src/Walsh_Hadamard/wh.c)

# RF_BT_RND backtracking random feature + WH feature
set(SOURCE_CODE_BT_RND ./src/RF_BT_RND/bt_rnd_util.cpp ./src/RF_BT_RND/bt_rnd_tree.cpp ./src/RF_BT_RND/bt_rnd_tree_node.cpp ./src/RF_BT_RND/bt_rnd_regressor.cpp ./src/RF_BT_RND/bt_rnd_regressor_builder.cpp)

# RND random number generator
set(SOURCE_CODE_RND ./src/RND/vnl_random.cxx ./src/RND/DTRandom.cpp)

set(SOURCE_CODE ${SOURCE_CODE1} ${SOURCE_CODE_OPENCV_UTIL} ${SOURCE_CODE_KABSCH} ${SOURCE_CODE_DTR} ${SOURCE_CODE_RND} ${SOURCE_CODE_DTC} ${SOURCE_CODE_BTDTR} ${SOURCE_CODE_BT_RND} ${SOURCE_CODE_WH})



add_executable(talker src/talker.cpp ${SOURCE_CODE})
target_link_libraries(talker ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${Boost_LIBRARIES} opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_calib3d flann)
add_dependencies(talker beginner_tutorials_generate_messages_cpp)
