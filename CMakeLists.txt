cmake_minimum_required(VERSION 3.0.2)
project(siamesenet)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    message_generation
)

add_service_files(
   FILES
   siameseSrv.srv
 )

generate_messages(
    DEPENDENCIES
    std_msgs
    sensor_msgs
)

catkin_package(
#    INCLUDE_DIRS include
    CATKIN_DEPENDS roscpp message_runtime std_msgs sensor_msgs
#    LIBRARIES siameseNet
#    DEPENDS
)
