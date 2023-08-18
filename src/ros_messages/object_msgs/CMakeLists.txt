cmake_minimum_required(VERSION 3.0.2)
project(object_msgs)

find_package(Eigen3 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
    std_msgs
    message_generation
    )

catkin_python_setup()

add_message_files(
    FILES
    MotionType.msg
    Pose32.msg
    Pose64.msg
    ObjectDimension.msg
    ObjectClass.msg
    ObjectClassEstimate.msg
    ObjectStateIndex.msg
    Object.msg
    Objects.msg
    ReferencePointType.msg
)

generate_messages(
    DEPENDENCIES
    std_msgs
)

catkin_package(
    INCLUDE_DIRS include ${EIGEN3_INCLUDE_DIRS}
    CATKIN_DEPENDS std_msgs message_runtime
    DEPENDS Eigen3
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)