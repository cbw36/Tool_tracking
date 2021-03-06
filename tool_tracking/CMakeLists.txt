cmake_minimum_required(VERSION 2.8.3)
project(tool_tracking)

find_package(catkin_simple REQUIRED)
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(catkin REQUIRED COMPONENTS message_generation std_msgs sensor_msgs geometry_msgs cwru_opencv_common tool_model vesselness_image_filter cwru_davinci_interface)

find_package(OpenCV REQUIRED)

catkin_simple()
#include_directories(include ${catkin_INCLUDE_DIRS} ${tool_model_INCLUDE_DIRS})
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

catkin_package(CATKIN_DEPENDS message_runtime std_msgs sensor_msgs geometry_msgs cwru_opencv_common tool_model)
# example boost usage
# find_package(Boost REQUIRED COMPONENTS system thread)

# C++0x support - not quite the same as final C++11!
# use carefully;  can interfere with point-cloud library
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pg -Q" )
# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg -Q")


# Libraries: uncomment the following and edit arguments to create a new library
# add_library(my_lib src/my_lib.cpp)   

  cs_add_library(tool_tracking_particle
              src/particle_filter.cpp
  )
  
  cs_add_library(tool_tracking_kalman
              src/kalman_filter.cpp
  )

cs_add_executable(tracking_particle src/tracking_particle.cpp)
cs_add_executable(tracking_kalman src/tracking_kalman.cpp)
#the following is required, if desire to link a node in this package with a library created in this same package
# edit the arguments to reference the named node and named library within this package
# target_link_library(example my_lib)
target_link_libraries(tool_tracking_particle tool_model_lib)
target_link_libraries(tool_tracking_kalman tool_model_lib)
target_link_libraries(tracking_particle tool_tracking_particle vesselness_image_filter_common_lib)
target_link_libraries(tracking_kalman tool_tracking_kalman vesselness_image_filter_common_lib)


cs_install()
cs_export()
