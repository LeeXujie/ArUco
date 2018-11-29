#include "rgbdreader.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include "filesystem.h"

using namespace std;

void print_binary(unsigned short input){
    unsigned short last_bit = 1;
    last_bit <<= 15;
    for(int i=0;i<16;i++){
        if((input&last_bit)>0)
            cout<<1;
        else
            cout<<0;
        input <<= 1;
    }
}

cv::Matx33d RGBDReader::getCamMat(){
    return rgb_cam_mat;
}

cv::Matx<double,5,1> RGBDReader::getDistCoeffs(){
    return rgb_dist_coeffs;
}

cv::Size RGBDReader::getImageSize(){
    return color_size;
}

double RGBDReader::get_depth_in_meters(char16_t input){
    return double(input)/1000.0;
}

double RGBDReader::get_time_stamp(std::string file_name){
    return stod(file_name.substr(0,file_name.find_first_of('.')));
}

void RGBDReader::find_correspondences(){
    set<string>::iterator closest_before=depth_file_names.begin();
    set<string>::iterator closest_after=depth_file_names.begin();

    for(string file_name: rgb_file_names){
        double rgb_ts=get_time_stamp(file_name);
        while(closest_after != depth_file_names.end()){
            if(get_time_stamp(*closest_after)<rgb_ts){
                closest_before = closest_after;
                closest_after ++;
            }
            else{
                break;
            }
        }

        if(closest_after == depth_file_names.end()){
            correspondences[file_name]=*closest_before;
            continue;
        }

        double closest_after_ts=get_time_stamp(*closest_after);
        double closest_before_ts=get_time_stamp(*closest_before);
        if( (closest_after_ts - rgb_ts) > (rgb_ts - closest_before_ts) )
            correspondences[file_name]=*closest_before;
        else
            correspondences[file_name]=*closest_after;
    }
    
    //fill correspondences_array for random access
    correspondences_array.clear();
    correspondences_array.reserve(correspondences.size());
    for(auto frame=correspondences.begin();frame!=correspondences.end();frame++)
        correspondences_array.push_back(frame);
    
}

bool RGBDReader::getCurrFrame(cv::Mat &depth, cv::Mat &color){
    curr_depth.copyTo(depth);
    curr_color.copyTo(color);
    return true;
}

void RGBDReader::getCurrColor(cv::Mat &color){
    curr_color.copyTo(color);
}

bool RGBDReader::readNextFrame(){
    if(curr_frame_num==-1){
        curr_frame=correspondences.begin();
    }
    else{
        curr_frame++;
    }
    curr_frame_num++;

    if(curr_frame==correspondences.end()){
        return false;
    }
    depth_registered=false;
    readCurrFrame();
    return true;
}

bool RGBDReader::getNextFrame(cv::Mat &depth, cv::Mat &color){
    if(!readNextFrame())
        return false;
    getCurrFrame(depth,color);
    return true;
}

bool RGBDReader::readFrameAt(unsigned long long frame_num){
    if(frame_num<correspondences.size()){
        curr_frame=correspondences_array[frame_num];
        curr_frame_num=frame_num;
    }
    else
        return false;

    readCurrFrame();
    return true;
}

bool RGBDReader::getFrameAt(unsigned long long frame_num, cv::Mat &depth, cv::Mat &color){
    if(!readFrameAt(frame_num))
        return false;
    getCurrFrame(depth,color);
    return true;
}

void RGBDReader::registerDepth(float max_dist){
    if(depth_registered)
        return;
    vector<cv::Point3f> point_3fs;
    point_3fs.reserve(depth_size.area());
    
    cv::Mat depth=undistort_depth(curr_depth);
    
    for(int r=0;r<depth_size.height;r++){
        char16_t *depth_row=depth.ptr<char16_t>(r);
        for(int c=0;c<depth_size.width;c++){
            double z=get_depth_in_meters(depth_row[c]);
            if(z==0 || z>max_dist)
                continue;

            cv::Point3d point_3d(c,r,1);
            cv::Point3f point_3f=depth_cam_mat.inv()*point_3d*z;
            point_3f=move_depth_to_color_coords(point_3f);
            point_3fs.push_back(point_3f);
        }
    }
    //project the points
    vector<cv::Point2f> projected_points;
    if(point_3fs.size()>0)
        cv::projectPoints(point_3fs,cv::Mat::zeros(3,1,CV_64FC1),cv::Mat::zeros(3,1,CV_64FC1),rgb_cam_mat,rgb_dist_coeffs,projected_points);

    vector<long long int> pixel_points(color_size.area(),-1);
    for(size_t i=0;i<projected_points.size();i++){
        int x = projected_points[i].x + .5;//rounding to the closest pixel
        int y = projected_points[i].y + .5;
        if( x < color_size.width && x >=0 && y < color_size.height && y>=0){//in range
            //get the color
            size_t index=y*color_size.width+x;
            if(pixel_points[index]<0){//if it is the first point for that pixel add the point for that pixel
                pixel_points[index]=i;
            }
            else if(point_3fs[i].z<point_3fs[pixel_points[index]].z){//if the z of the point is smaller than a previous point replace the previous point
                pixel_points[index]=i;
            }
        }
    }
    curr_points.clear();
    curr_projected_points.clear();
    for(size_t i=0;i<pixel_points.size();i++)
        if(pixel_points[i]>=0){
            curr_points.push_back(point_3fs[pixel_points[i]]);
            curr_projected_points.push_back(projected_points[pixel_points[i]]);
        }
}

bool RGBDReader::getXYZImage(cv::Mat &xyz, cv::Mat transform, float max_dist){
    if(curr_depth.empty())
        return false;
    registerDepth(max_dist);
    xyz=cv::Mat::zeros(depth_size,CV_32FC3);
    //make fast access array
    vector<cv::Vec3f*> xyz_fast;
    for(int r=0;r<xyz.rows;r++)
        xyz_fast.push_back(xyz.ptr<cv::Vec3f>(r));
    //
    for(size_t i=0;i<curr_projected_points.size();i++){
        int x = curr_projected_points[i].x + .5;//rounding to the closest pixel
        int y = curr_projected_points[i].y + .5;
        cv::Point3f point_3D=curr_points[i];
        if(!transform.empty())
            point_3D=Transform3d(transform)(point_3D);
        xyz_fast[y][x]=point_3D;
    }
    return true;
}

bool RGBDReader::getCurrCloud(std::vector<cv::Point3f> &points,std::vector<cv::Vec3b> &colors,Transform3d T, float max_dist){
    if(curr_depth.empty())
        return false;
    registerDepth(max_dist);
    
    cv::Mat rgb;
    cv::cvtColor(curr_color,rgb,CV_BGR2RGB);
    for(cv::Point3f point:curr_points)
        points.push_back(T(point));
    
    //make fast access arrays
    vector<cv::Vec3b*> rgb_fast;
    for(int r=0;r<rgb.rows;r++)
        rgb_fast.push_back(rgb.ptr<cv::Vec3b>(r));

    for(size_t i=0;i<curr_projected_points.size();i++){
        int x = curr_projected_points[i].x + .5;//rounding to the closest pixel
        int y = curr_projected_points[i].y + .5;
        colors.push_back(rgb_fast[y][x]);
    }

    return true;
}



