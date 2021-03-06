/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016 Case Western Reserve University
 *
 *	 Ran Hao <rxh349@case.edu>
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *	 notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *	 copyright notice, this list of conditions and the following
 *	 disclaimer in the documentation and/or other materials provided
 *	 with the distribution.
 *   * Neither the name of Case Western Reserve University, nor the names of its
 *	 contributors may be used to endorse or promote products derived
 *	 from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PARTICLEFILTER_H
#define PARTICLEFILTER_H

#include <vector>
#include <stdio.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <stdlib.h>
#include <math.h>

#include <string>
#include <cstring>

#include <tool_model_lib/tool_model.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/image_encodings.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include <vesselness_image_filter_cpu/vesselness_lib.h>
#include <boost/random/normal_distribution.hpp>

#include <geometry_msgs/Transform.h>
#include <cwru_davinci_interface/davinci_interface.h>

#include <tf/transform_listener.h>
#include <cwru_davinci_interface/davinci_interface.h>
#include <cwru_davinci_kinematics/davinci_kinematics.h>

//#include <cwru_xform_utils/xform_utils.h> //jade (surgical31)
#include <xform_utils/xform_utils.h> //indigo

class ParticleFilter {

private:
	cv::Mat Cam_left;//Gcb from left camera
	cv::Mat Cam_right;//Gcb for right camera

	ros::NodeHandle node_handle;

	ToolModel newToolModel;//updated configuration
	ToolModel::toolModel predicted_real_pose; //predicted tool pose from FK updated at every time step

	unsigned int numParticles; //total number of particles

    cv::Mat toolImage_left_arm_1; //image in left camera frame.  Will store each particle vs segmented image to calculate chamfer distance
    cv::Mat toolImage_right_arm_1; //image in right camera frame.  Will store each particle vs segmented image to calculate chamfer distance
    cv::Mat toolImage_left_arm_2; //Not currently used, but same as above for arm 2
    cv::Mat toolImage_right_arm_2; //Not currently used but same as above for arm 2

    cv::Mat toolImage_left_temp; //composite of all rendered particles in left camera
    cv::Mat toolImage_right_temp; //composite of all rendered particles in right camera

	std::vector<double> matchingScores_arm_1; // particle scores (matching scores)
	std::vector<double> matchingScores_arm_2; // particle scores (matching scores)

    // particles for arm 1/2 stored as a vector of toolModels
    // toolModel= 3x1 cv::Mats tvec_cyl, rvec_cyl, tvec_elp, rvec_elp, tvec_grip1(2), rvec_grip1(2)
	std::vector<std::vector <double> > particles_arm_1; // particles
	std::vector<double> particleWeights_arm_1; // particle weights calculated from matching scores
	std::vector<ToolModel::toolModel> particles_arm_2; // particles
	std::vector<double> particleWeights_arm_2; // particle weights calculated from matching scores

    //Intermediary representations of G_cb. Will be converted to cv::mat
    Eigen::Affine3d arm_1__cam_l;
    Eigen::Affine3d arm_2__cam_l;
    Eigen::Affine3d arm_1__cam_r;
    Eigen::Affine3d arm_2__cam_r;

    cv::Mat Cam_left_arm_1; //Gcb for left camera and arm 1
    cv::Mat Cam_right_arm_1; //Gcb for right camera and arm 1
    cv::Mat Cam_left_arm_2; //Gcb for left camera and arm 2
    cv::Mat Cam_right_arm_2; //Gcb for right camera and arm 2

    Davinci_fwd_solver kinematics; // FK for davinci class

    std::vector<double> sensor_1; //sensor readings from davinci of joint angles 7x1
    std::vector<double> sensor_2; //sensor readings from davinci of joint angles 7x1

    /**
     * @brief get P_right from subscriber
     * @param projectionRight
     */
    void projectionRightCB(const sensor_msgs::CameraInfo::ConstPtr &projectionRight);

    /**
     * @brief get P_left from subscriber
     * @param projectionLeft
     */
    void projectionLeftCB(const sensor_msgs::CameraInfo::ConstPtr &projectionLeft);

    ros::Subscriber projectionMat_subscriber_r; //Get P_right by subscribing to camera_info
    ros::Subscriber projectionMat_subscriber_l; //Get P_right by subscribing to camera_info

    bool freshCameraInfo; //Determine if a new image to analyze

	double t_step;
	double t_1_step;

    double down_sample_rate; //annealing rate.  High at start to allow large perturbations and decrease as we converge
    double error;

public:

    cv::Mat raw_image_left; //left rendered Image
    cv::Mat raw_image_right; //right rendered Image

    cv::Mat P_left; //Projection matrix for left camera
    cv::Mat P_right; //Projection matrix for right camera

    /**
    * @brief - The default constructor
    * Call initializeParticles()
    * Store Gcb from testing as a cv::mat
    * Initialize empty tool images
    */
	ParticleFilter(ros::NodeHandle *nodehandle);

    /**
     * @brief- The deconstructor
     * Not currently used
     */
	~ParticleFilter();

    /**
     * @brief- initializes the particles with the following actions:
     * 		Initialize matchingScores_arm_x, particles_arm_x and particle_weights_arm_x arrays
     * 		Compute an initial guess and generate particles around it with getCoarseGuess()
     */
	void initializeParticles();

    /**
     * @brief- Compute an initial guess from FK and generate particles around it
     * Used when particles initialized and there is no prior particle data
     * First get sensor info from all 7 joints
     * Then compute tvec_cyl/rvec_cyl from FK of joints 1-4
     * Then generate particles randomly about tvec_cyl/rvec_cyl/joints 5,6,7
     * Then update toolModel to include joints 5,6,7
     */
    void getCoarseGuess();

	/**
	 * @brief Main tracking function
	 * @param segmented_left : segmented image for left camera
	 * @param segmented_right : segmented image for right camera
	 * @return
	 */
	void trackingTool(const cv::Mat &segmented_left, const cv::Mat &segmented_right);

	/**
	 * @brief low variance resampling
	 * @param sampleModel : input particles
	 * @param particleWeight : input normalized weights
	 * @param update_particles : output particles
	 */
	void resamplingParticles(const std::vector< std::vector<double> > &sampleModel,
							 const std::vector<double> &particleWeight,
							 std::vector<std::vector<double> > &update_particles);

	/**
	 * @brief Compute matching score between a particle and a segmented image; p(z_t|x_t)
	 * Render the particle and segmented image together and compute the chamfer distance between them
	 * Used in measurement model
	 * @param toolImage_left image of left segmented image and particle transformed under left camera's frame
	 * @param toolImage_right image of right segmented image and particle transformed under right camera's frame
	 * @param toolPose toolModel representation of a particle (to compare with segmented images)
	 * @param segmented_left segmented image from left camera (to compare with particle)
	 * @param segmented_right segmented image from right camera (to compare with particle)
	 * @param Cam_left left camera transform used to transform particle
	 * @param Cam_right right camera transform used to transform particle
	 * @return chamfer score between particle and segmented images
	 */
	double measureFuncSameCam(cv::Mat & toolImage_left, cv::Mat & toolImage_right, ToolModel::toolModel &toolPose,
							  const cv::Mat &segmented_left, const cv::Mat &segmented_right, cv::Mat &Cam_left, cv::Mat &Cam_right);
	/**
	 * @brief Motion model, propagate the particles using velocity computed from joint sensors
	 * @param best_particle_last: last time step best particle, used to compute the nominal velocity
	 * @param updatedParticles : input and output particles
	 */
	void updateParticles(std::vector <double> &best_particle_last, double &maxScore, std::vector<std::vector <double> > &updatedParticles, ToolModel::toolModel & predicted_real_pose);

	/**
	 * @brief Initialize a set of noised particles from the FK guess of initial particle
	 * Used only on initialization
	 * @param inputParticle Initial 19x1 particle from FK and sensor info
	 * @param noisedParticles A set of particles randomly perturbed from input
	 */
	void computeNoisedParticles(std::vector <double> & inputParticle, std::vector< std::vector <double> > & noisedParticles);

	/**
 	* @brief compute the error between the predicted pose from FK and the best particles guess of the pose
	 * Not currently used (would be used in updateParticles)
 	* @param real_pose pose from Gazebo
 	* @param bestParticle
 	*/
	void showGazeboToolError(ToolModel::toolModel &real_pose, ToolModel::toolModel &bestParticle);

	/**
	 * @brief convert a toolModel representation to a 24x1 matrix representation
	 * Not currently used; would be used to find the error in simulation (from showGazeboToolError)
	 * @param inputToolModel
	 * @param toolMatrix output matrix representation
	 */
	void convertToolModeltoMatrix(const ToolModel::toolModel &inputToolModel, cv::Mat &toolMatrix);

	/**
 	* @brief Convert a rotation matrix to a rotation vector
	* Used when converting sensor readings to a rotation vector in FK
 	* @param trans - input SE(3)
 	* @param rot_vec output 3x1 cv::mat
 	*/
	void computeRodriguesVec(const Eigen::Affine3d & trans, cv::Mat rot_vec);

	/**
	 * @brief convert from an SE(3) representation to a cv::Mat representation
	 * Not currently use
	 * @param trans
	 * @param outputMatrix output cv::Mat
	 */
	void convertEigenToMat(const Eigen::Affine3d & trans, cv::Mat & outputMatrix);
};

#endif
