cmake_minimum_required(VERSION 3.0.2)
project(keypoint_msg)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  message_generation
)

add_message_files(
  FILES
  KeypointArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS
  roscpp
  std_msgs
  geometry_msgs
  sensor_msgs
  message_runtime
)
