#ifndef GLOBAL_ICP_H
#define GLOBAL_ICP_H
#include "basictypes/picoflann.h"
#include <opencv2/core.hpp>
#include <random>

class Global_ICP
{
    struct picoflann_point3f_adaptor{
        inline float operator()(const cv::Point3f &input, int &dim) const{
            return cv::Vec3f(input)[dim];
        }
        inline float operator()(const cv::Point3f &input, const uint16_t &dim) const{
            return cv::Vec3f(input)[dim];
        }
    };
    std::vector<picoflann::KdTreeIndex<3,picoflann_point3f_adaptor>> trees;
    std::vector<std::vector<cv::Point3f>> clouds;
    std::vector<std::vector<cv::Mat>> transforms;
    size_t num_clouds=0;
public:
    Global_ICP(const std::vector<std::vector<cv::Point3f>> &clouds, float subsample_radius=0.01);
    Global_ICP(const std::vector<cv::Mat> &in_depthmaps, float subsample_radius=0.01);
    void initialize(float subsample_radius);
    void execute(std::pair<double,double> radius_range=std::pair<double,double>(0.01,0.05), int num_iterations=10);
    std::vector<std::vector<cv::Point3f>> getClouds(const std::vector<std::vector<cv::Point3f>> &in_clouds, int num_iterations=-1);
    std::vector<std::vector<cv::Mat>> getTransforms();

    static void transformPointcloud(std::vector<cv::Point3f> &cloud, cv::Mat T);
    static std::vector<cv::Point3f> subsamplePointcloud(const std::vector<cv::Point3f> &in_cloud, float radius, unsigned int seed);
    int importance(double distance,double max_distance,int iteration,int num_iterations);
    template<typename T>
    void subsampleDepthmap2Pointcloud(cv::Mat &depthmap,std::vector<cv::Point3f> &cloud, float subsample_radius);
};

#endif // GLOBAL_ICP_H
