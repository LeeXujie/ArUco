#ifndef RGBDREADER_H
#define RGBDREADER_H

#include <opencv2/core.hpp>
#include <memory>
#include <map>
#include <set>
#include <transform3d.h>

//undistortion is suppposed to be done internally

class RGBDReader
{

    virtual cv::Point3f move_depth_to_color_coords(cv::Point3f)=0;
    virtual cv::Mat undistort_depth(cv::Mat)=0;
    virtual cv::Mat undistort_color(cv::Mat)=0;
    virtual double get_depth_in_meters(char16_t);
    virtual double get_time_stamp(std::string file_name);
    virtual bool readCurrFrame()=0;

protected:

    cv::Matx33d rgb_cam_mat;
    cv::Matx<double,5,1> rgb_dist_coeffs;
    cv::Mat rgb_undistort_map;
    cv::Matx33d depth_cam_mat;
    cv::Matx<double,5,1> depth_dist_coeffs;
    cv::Mat depth_undistort_map;
    Transform3d depth2rgb_transformer;
    cv::Mat curr_depth;
    cv::Mat curr_color;
    std::map<std::string,std::string> correspondences;
    std::vector<std::map<std::string,std::string>::iterator> correspondences_array;
    std::map<std::string,std::string>::iterator curr_frame;
    std::set<std::string> rgb_file_names;
    std::set<std::string> depth_file_names;
    //a function to find the assosiaction between the depth and RGB frames using their time stamps
    void find_correspondences();
    long long int curr_frame_num=-1;
    std::string dir_path;
    cv::Size depth_size,color_size;
    std::vector<cv::Point3f> curr_points;
    std::vector<cv::Point2f> curr_projected_points;
    bool depth_registered;
    
public:
    
    virtual bool getNextFrame(cv::Mat &depth, cv::Mat &color);
    virtual bool readFrameAt(unsigned long long frame_num);
    bool getFrameAt(unsigned long long frame_num, cv::Mat &depth, cv::Mat &color);
    bool readNextFrame();
    bool getCurrFrame(cv::Mat &depth, cv::Mat &color);
    bool getCurrCloud(std::vector<cv::Point3f> &points, std::vector<cv::Vec3b> &colors, Transform3d T=Transform3d(), float max_dist=2);
    cv::Matx33d getCamMat();
    cv::Matx<double,5,1> getDistCoeffs();
    cv::Size getImageSize();
    void registerDepth(float max_dist=2);
    bool getXYZImage(cv::Mat&, const cv::Mat transform=cv::Mat(), float max_dist=2);
    void getCurrColor(cv::Mat &color);

};

#endif // RGBDREADER_H
