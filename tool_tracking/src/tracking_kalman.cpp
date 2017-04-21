#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <cwru_opencv_common/projective_geometry.h>
#include <tool_tracking/kalman_filter.h>

bool freshImage;
bool freshCameraInfo;
bool freshVelocity;

using namespace std;
using namespace cv_projective;

std::vector<cv::Mat> trackingImgs;  ///this should be CV_32F

void newImageCallback(const sensor_msgs::ImageConstPtr &msg, cv::Mat *outputImage) {
	cv_bridge::CvImagePtr cv_ptr;
	try {
		//cv::Mat src =  cv_bridge::toCvShare(msg,"32FC1")->image;
		//outputImage[0] = src.clone();
		cv_ptr = cv_bridge::toCvCopy(msg);
		outputImage[0] = cv_ptr->image;
		freshImage = true;
	}
	catch (cv_bridge::Exception &e) {
		ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
	}
}

cv::Mat segmentation(cv::Mat &InputImg) {

	cv::Mat src, src_gray;
	cv::Mat grad;

	cv::Mat res;
	src = InputImg;

	cv::resize(src, src, cv::Size(), 1, 1);

	double lowThresh = 43;

	cv::cvtColor(src, src_gray, CV_BGR2GRAY);

	cv::blur(src_gray, src_gray, cv::Size(3, 3));

	cv::Canny(src_gray, grad, lowThresh, 4 * lowThresh, 3); //use Canny segmentation

	grad.convertTo(res, CV_32FC1);

	return res;

}

int main(int argc, char **argv) {

	ros::init(argc, argv, "tracking_node");

	ros::NodeHandle nh;
	/******  initialization  ******/
	KalmanFilter UKF(&nh);

	freshCameraInfo = false;
	freshImage = false;
	//freshVelocity = false;//Moving all velocity-related things inside of the kalman.

	cv::Mat seg_left  = cv::Mat::zeros(480, 640, CV_32FC1);
	cv::Mat seg_right  = cv::Mat::zeros(480, 640, CV_32FC1);

	trackingImgs.resize(2);

	//TODO: get image size from camera model, or initialize segmented images,

	cv::Mat rawImage_left = cv::Mat::zeros(480, 640, CV_32FC1);
	cv::Mat rawImage_right = cv::Mat::zeros(480, 640, CV_32FC1);

	image_transport::ImageTransport it(nh);
	image_transport::Subscriber img_sub_l = it.subscribe(
		"/davinci_endo/left/image_raw",
		1,
		boost::function<void(const sensor_msgs::ImageConstPtr &)>(boost::bind(newImageCallback, _1, &rawImage_left))
	);
	image_transport::Subscriber img_sub_r = it.subscribe(
		"/davinci_endo/right/image_raw",
		1,
		boost::function<void(const sensor_msgs::ImageConstPtr &)>(boost::bind(newImageCallback, _1, &rawImage_right))
	);

	ROS_INFO("---- done subscribe -----");

	/*** Timer set up ***/
	ros::Rate loop_rate(50);

	ros::Duration(2).sleep();

	while (nh.ok()) {
		ros::spinOnce();

		if (freshImage ){

			UKF.tool_rawImg_left = rawImage_left.clone();
			UKF.tool_rawImg_right = rawImage_right.clone();

			seg_left = segmentation(rawImage_left);
			seg_right = segmentation(rawImage_right);
			//ROS_INFO("AFTER SEG");
//			cv::imshow("Cam L", rawImage_left);
//			cv::imshow("Cam R", rawImage_right);`
//			cv::imshow("Seg L", seg_left);
//			cv::imshow("Seg R", seg_right);
//			cv::waitKey(10);

            UKF.UKF_double_arm(seg_left, seg_right);
			//UKF.measureFunc(UKF.toolImage_left_arm_1, UKF.toolImage_right_arm_1, initial_test, seg_left, seg_right, UKF.Cam_left_arm_1, UKF.Cam_right_arm_1);

			freshImage = false;
            //ros::Duration(3).sleep();
		}

		//We want to update our filter whenever the robot is doing anything, not just when we are getting images.

        //	ToolModel::toolModel currentToolModel;
        //	convertToolModel(current_mu, currentToolModel,1);
        //	measureFunc(currentToolModel, segmented_left, segmented_right, zt);

		loop_rate.sleep();  //or cv::waitKey(10);
	}
}
