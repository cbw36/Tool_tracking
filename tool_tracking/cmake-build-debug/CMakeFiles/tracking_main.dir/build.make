# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/ranhao/Documents/clion-2016.3.3/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/ranhao/Documents/clion-2016.3.3/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ranhao/ros_ws/src/Tool_tracking/tool_tracking

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ranhao/ros_ws/src/Tool_tracking/tool_tracking/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/tracking_main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tracking_main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tracking_main.dir/flags.make

CMakeFiles/tracking_main.dir/src/tracking_main.cpp.o: CMakeFiles/tracking_main.dir/flags.make
CMakeFiles/tracking_main.dir/src/tracking_main.cpp.o: ../src/tracking_main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ranhao/ros_ws/src/Tool_tracking/tool_tracking/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tracking_main.dir/src/tracking_main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tracking_main.dir/src/tracking_main.cpp.o -c /home/ranhao/ros_ws/src/Tool_tracking/tool_tracking/src/tracking_main.cpp

CMakeFiles/tracking_main.dir/src/tracking_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tracking_main.dir/src/tracking_main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ranhao/ros_ws/src/Tool_tracking/tool_tracking/src/tracking_main.cpp > CMakeFiles/tracking_main.dir/src/tracking_main.cpp.i

CMakeFiles/tracking_main.dir/src/tracking_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tracking_main.dir/src/tracking_main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ranhao/ros_ws/src/Tool_tracking/tool_tracking/src/tracking_main.cpp -o CMakeFiles/tracking_main.dir/src/tracking_main.cpp.s

CMakeFiles/tracking_main.dir/src/tracking_main.cpp.o.requires:

.PHONY : CMakeFiles/tracking_main.dir/src/tracking_main.cpp.o.requires

CMakeFiles/tracking_main.dir/src/tracking_main.cpp.o.provides: CMakeFiles/tracking_main.dir/src/tracking_main.cpp.o.requires
	$(MAKE) -f CMakeFiles/tracking_main.dir/build.make CMakeFiles/tracking_main.dir/src/tracking_main.cpp.o.provides.build
.PHONY : CMakeFiles/tracking_main.dir/src/tracking_main.cpp.o.provides

CMakeFiles/tracking_main.dir/src/tracking_main.cpp.o.provides.build: CMakeFiles/tracking_main.dir/src/tracking_main.cpp.o


# Object files for target tracking_main
tracking_main_OBJECTS = \
"CMakeFiles/tracking_main.dir/src/tracking_main.cpp.o"

# External object files for target tracking_main
tracking_main_EXTERNAL_OBJECTS =

devel/lib/tool_tracking/tracking_main: CMakeFiles/tracking_main.dir/src/tracking_main.cpp.o
devel/lib/tool_tracking/tracking_main: CMakeFiles/tracking_main.dir/build.make
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libtool_model_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libopencv_ui_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libcircle_detection_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libblock_detection_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libgrab_cut_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libprojective_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libopencv_3d_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libopencv_2d_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libopencv_local_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libopencv_rot_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libcolor_model_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libvesselness_lib.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/libdynamic_reconfigure_config_init_mutex.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/libimage_transport.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/libmessage_filters.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libtinyxml.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/libclass_loader.so
devel/lib/tool_tracking/tracking_main: /usr/lib/libPocoFoundation.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libdl.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/libroscpp.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libboost_signals.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/libxmlrpcpp.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/libroslib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libcv_bridge.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/librosconsole.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/librosconsole_log4cxx.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/librosconsole_backend_interface.so
devel/lib/tool_tracking/tracking_main: /usr/lib/liblog4cxx.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libboost_regex.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/libroscpp_serialization.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/librostime.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/libcpp_common.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libboost_system.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libboost_thread.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libpthread.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libdavinci_interface.so
devel/lib/tool_tracking/tracking_main: devel/lib/libtool_tracking.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libtool_model_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libopencv_ui_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libcircle_detection_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libblock_detection_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libgrab_cut_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libprojective_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libopencv_3d_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libopencv_2d_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libopencv_local_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libopencv_rot_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libcolor_model_lib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libvesselness_lib.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/libdynamic_reconfigure_config_init_mutex.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/libimage_transport.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/libmessage_filters.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libtinyxml.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/libclass_loader.so
devel/lib/tool_tracking/tracking_main: /usr/lib/libPocoFoundation.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libdl.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/libroscpp.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libboost_signals.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/libxmlrpcpp.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/libroslib.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libcv_bridge.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/librosconsole.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/librosconsole_log4cxx.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/librosconsole_backend_interface.so
devel/lib/tool_tracking/tracking_main: /usr/lib/liblog4cxx.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libboost_regex.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/libroscpp_serialization.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/librostime.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
devel/lib/tool_tracking/tracking_main: /opt/ros/indigo/lib/libcpp_common.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libboost_system.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libboost_thread.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libpthread.so
devel/lib/tool_tracking/tracking_main: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so
devel/lib/tool_tracking/tracking_main: /home/ranhao/ros_ws/devel/lib/libdavinci_interface.so
devel/lib/tool_tracking/tracking_main: CMakeFiles/tracking_main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ranhao/ros_ws/src/Tool_tracking/tool_tracking/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable devel/lib/tool_tracking/tracking_main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tracking_main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tracking_main.dir/build: devel/lib/tool_tracking/tracking_main

.PHONY : CMakeFiles/tracking_main.dir/build

CMakeFiles/tracking_main.dir/requires: CMakeFiles/tracking_main.dir/src/tracking_main.cpp.o.requires

.PHONY : CMakeFiles/tracking_main.dir/requires

CMakeFiles/tracking_main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tracking_main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tracking_main.dir/clean

CMakeFiles/tracking_main.dir/depend:
	cd /home/ranhao/ros_ws/src/Tool_tracking/tool_tracking/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ranhao/ros_ws/src/Tool_tracking/tool_tracking /home/ranhao/ros_ws/src/Tool_tracking/tool_tracking /home/ranhao/ros_ws/src/Tool_tracking/tool_tracking/cmake-build-debug /home/ranhao/ros_ws/src/Tool_tracking/tool_tracking/cmake-build-debug /home/ranhao/ros_ws/src/Tool_tracking/tool_tracking/cmake-build-debug/CMakeFiles/tracking_main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tracking_main.dir/depend

