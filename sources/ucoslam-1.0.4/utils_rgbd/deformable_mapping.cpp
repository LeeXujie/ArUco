#include <iostream>
#include "map.h"
#include <cmath>
#include "rgbdreaderfactory.h"
#include "sequenceoptimizer.h"
#include "sparselevmarq.h"
#include <limits>
#include <depthmaps/embeddedregistration.h>
#include <depthmaps/icp.h>
#include <cstdio>
#include <depthmaps/depthmaputils.h>
#include <depthmaps/depthmapsetutils.h>
#include <depthmaps/pointcloudutils.h>
#include <depthmaps/pclutils.h>
#include "global_icp.h"

using namespace std;

cv::Mat get_axis_angle(cv::Vec3f cam_dir) {
    cv::Vec3d z(0,0,1);
    cv::Vec3d dir(cam_dir);
    cv::Vec3d axis;
    if(dir==z)
        axis=z;
    else
        axis=z.cross(dir);
    double theta = std::acos(z.dot(dir));
//    cout<<"cam_dir"<<dir<<endl;
//    cout<<"axis"<<axis<<endl;
    axis/=cv::norm(axis);
    axis*=theta;
    return cv::Mat(axis);
}

cv::Mat getMatrix(double qx,double qy, double qz,double qw,double tx,double ty ,double tz) { //from rafa


    double qx2 = qx*qx;
    double qy2 = qy*qy;
    double qz2 = qz*qz;


    cv::Mat m=cv::Mat::eye(4,4,CV_64F);

    m.at<double>(0,0)=1 - 2*qy2 - 2*qz2;
    m.at<double>(0,1)=2*qx*qy - 2*qz*qw;
    m.at<double>(0,2)=2*qx*qz + 2*qy*qw;
    m.at<double>(0,3)=tx;

    m.at<double>(1,0)=2*qx*qy + 2*qz*qw;
    m.at<double>(1,1)=1 - 2*qx2 - 2*qz2;
    m.at<double>(1,2)=2*qy*qz - 2*qx*qw;
    m.at<double>(1,3)=ty;

    m.at<double>(2,0)=2*qx*qz - 2*qy*qw;
    m.at<double>(2,1)=2*qy*qz + 2*qx*qw;
    m.at<double>(2,2)=1 - 2*qx2 - 2*qy2;
    m.at<double>(2,3)=tz;
    return m;
}

void read_orbslam2_keyframes(string file_path,map<unsigned long, unsigned long> &frame2keyframe_index, map<unsigned long, cv::Mat> &orbslam2_pose_converters, unsigned long &last_frame_index) {
    frame2keyframe_index.clear();
    orbslam2_pose_converters.clear();
    std::ifstream input_file(file_path);
    double frame_num,x,y,z,qx,qy,qz,qw;
    int key_frame_index=0;
    while(input_file>>frame_num>>x>>y>>z>>qx>>qy>>qz>>qw) {
        x/=1000.0;
        y/=1000.0;
        z/=1000.0;
        frame2keyframe_index[frame_num]=key_frame_index++;
        orbslam2_pose_converters[frame_num]=getMatrix(qx,qy,qz,qw,x,y,z);
        last_frame_index=frame_num;
    }
}

void draw_3d_line_points(cv::Point3f start, cv::Point3f end, int steps, cv::Vec3b start_color, cv::Vec3b end_color, vector<cv::Point3f> &point_cloud, vector<cv::Vec3b> &colors) {
    cv::Point3d step=(end-start)/steps;
    cv::Vec3d color_step=(cv::Vec3d(end_color)-cv::Vec3d(start_color))/double(steps);
    cv::Point3d point=start;
    for(int i=0; i<steps; i++,point+=step) {
        point_cloud.push_back(point);
        colors.push_back(start_color+cv::Vec3b(i*color_step));
    }
}

void write_pcd(string path, cv::Mat &xyz, cv::Mat &color_image) {
    ofstream of(path,ios_base::binary);
    if(!of.is_open())
        throw std::runtime_error("Could not open a file to write at: "+path);
    cout<<xyz.cols*xyz.rows<<" points to be written."<<endl;
    of<<"# .PCD v0.7 - Point Cloud Data File Format"<<endl;
    of<<"VERSION 0.7"<<endl;
    of<<"FIELDS x y z rgb"<<endl;
    of<<"SIZE 4 4 4 4"<<endl;
    of<<"TYPE F F F F"<<endl;
    of<<"COUNT 1 1 1 1"<<endl;
    of<<"WIDTH "<<xyz.cols<<endl;
    of<<"HEIGHT "<<xyz.rows<<endl;
    of<<"VIEWPOINT 0 0 0 1 0 0 0"<<endl;
    of<<"POINTS "<<xyz.cols*xyz.rows<<endl;
    of<<"DATA binary"<<endl;
    //write the points in binary form
    for(size_t r=0; r<xyz.rows; r++){
        cv::Vec3f *xyz_row=xyz.ptr<cv::Vec3f>(r);
        cv::Vec3b *color_row=color_image.ptr<cv::Vec3b>(r);
        for(size_t c=0; c<xyz.cols; c++){
            cv::Vec3f point=xyz_row[c];
            cv::Vec3b colors=color_row[c];
            char valid=1;
            if(point[0]==0 && point[1]==0 && point[2]==0){
                point[0]=point[1]=point[2]=NAN;
                valid=0;
            }

            for(int i=0; i<3; i++) {
                float value=point[i];
                of.write((char*)(&value),sizeof value);
            }
            char32_t color=0;
            for(int i=0; i<3; i++) {
                color <<= 8;
                color += char32_t(colors[i]);
            }
            color <<= 8;
            color += char32_t(valid);

            if(std::isnan(point[0]))
                of.write((char*)(&point[0]),sizeof point[0]);
            else
                of.write((char*)(&color),sizeof color);
        }
    }
}

void write_pcd(string path, vector<cv::Point3f> &points, vector<cv::Vec3b> &colors) {
    ofstream of(path,ios_base::binary);
    if(!of.is_open())
        throw std::runtime_error("Could not open a file to write at: "+path);
    cout<<points.size()<<" points to be written."<<endl;
    cout<<colors.size()<<endl;
    of<<"# .PCD v0.7 - Point Cloud Data File Format"<<endl;
    of<<"VERSION 0.7"<<endl;
    of<<"FIELDS x y z rgb"<<endl;
    of<<"SIZE 4 4 4 4"<<endl;
    of<<"TYPE F F F F"<<endl;
    of<<"COUNT 1 1 1 1"<<endl;
    of<<"WIDTH "<<points.size()<<endl;
    of<<"HEIGHT 1"<<endl;
    of<<"VIEWPOINT 0 0 0 1 0 0 0"<<endl;
    of<<"POINTS "<<points.size()<<endl;
    of<<"DATA binary"<<endl;
    //write the points in binary form
    for(size_t i=0; i<points.size(); i++) {
        cv::Vec3f point(points[i]);
        for(int j=0; j<3; j++) {
            float value=point[j];
            of.write((char*)(&value),sizeof value);
        }
        char32_t color=0;
        for(int j=0; j<3; j++) {
            color <<= 8;
            color += char32_t(colors[i][j]);
        }
        of.write((char*)(&color),sizeof color);
    }
}

vector<map<unsigned int,cv::Point3f>> mappoint_keyframe_keypoint_coords;
vector<cv::Point3f> mappoint_coords;
map<unsigned int, cv::Mat> keyframe_transforms;
vector<pair<unsigned int, unsigned int>> keyframe2frame_index;
map<unsigned int, unsigned int> keyframe_io_index;

void getTransformsPointCloud(map<unsigned long int, unsigned long int> frame2keyframe_index,shared_ptr<RGBDReader> rgbd_reader,vector<cv::Point3f> &points,vector<cv::Vec3b> &colors,map<unsigned int,cv::Mat>& transforms){
    points.clear();
    colors.clear();
    for(auto &frame2keyframe: frame2keyframe_index){
        unsigned int frame_index=frame2keyframe.first;
        unsigned int keyframe_index=frame2keyframe.second;

        cout<<frame_index<<endl;
//        cout<<transforms[keyframe_index]<<endl;
        rgbd_reader->readFrameAt(frame_index);
        rgbd_reader->getCurrCloud(points,colors,transforms[keyframe_index]);
    }
}

void keyframe_transforms_to_io_vec(ucoslam::SparseLevMarq<double>::eVector &io_vec){
    io_vec.resize(keyframe_transforms.size()*6);
    unsigned int curr_index=0;
    for(std::pair<const unsigned int,cv::Mat> &keyframe_transform:keyframe_transforms){
        cv::Mat transform=keyframe_transform.second;
        cv::Mat r;
        cv::Rodrigues(transform(cv::Range(0,3),cv::Range(0,3)),r);
        for(int i=0;i<3;i++){
            io_vec(curr_index+i)=r.at<float>(i);
            io_vec(curr_index+i+3)=transform.at<float>(i,3);
        }
        curr_index+=6;
    }
}

void io_vec_to_keyframe_transforms(ucoslam::SparseLevMarq<double>::eVector &io_vec){
    unsigned int curr_index=0;
    for(std::pair<const unsigned int,cv::Mat> &keyframe_transform:keyframe_transforms){
        cv::Mat transform=cv::Mat::eye(4,4,CV_32FC1);
        cv::Mat r(3,1,CV_32FC1);
        for(int i=0;i<3;i++){
            r.at<float>(i)=io_vec(curr_index+i);
            transform.at<float>(i,3)=io_vec(curr_index+i+3);
        }
        cv::Rodrigues(r,transform(cv::Range(0,3),cv::Range(0,3)));
        keyframe_transform.second=transform;
        curr_index+=6;
    }
}

void error_function(const typename ucoslam::SparseLevMarq<double>::eVector &input, typename ucoslam::SparseLevMarq<double>::eVector &error){

    vector<vector<vector<double>>> mappoint_diffs(mappoint_keyframe_keypoint_coords.size());
    unsigned int mappoint_index=0;
    unsigned int total_errors=0;
    for(auto &keyframe_keypoint_coords:mappoint_keyframe_keypoint_coords){//loop over each mappoint
        vector<cv::Point3f> transformed_points;
        cv::Point3f sum(0,0,0);
        mappoint_diffs[mappoint_index].resize(keyframe_keypoint_coords.size());
        unsigned int keyframe_index=0;
        for(auto &keypoint_coords:keyframe_keypoint_coords){//loop over each corresponding keyframe
            unsigned int io_index=keyframe_io_index[keypoint_coords.first];
            cv::Vec3f coords=keypoint_coords.second;
            cv::Vec3f r,t;
            cv::Matx33f R;
            for(int i=0;i<3;i++){
                r(i)=input(io_index+i);
                t(i)=input(io_index+3+i);
            }
            cv::Rodrigues(r,R);
            cv::Point3f transformed_coords=R*coords+t;
            sum+=transformed_coords;
            transformed_points.push_back(transformed_coords);
        }
        cv::Point3f average=sum/double(transformed_points.size());
        for(size_t i=0;i<transformed_points.size();i++){
            mappoint_diffs[mappoint_index][keyframe_index].push_back(transformed_points[i].x-average.x);
            mappoint_diffs[mappoint_index][keyframe_index].push_back(transformed_points[i].y-average.y);
            mappoint_diffs[mappoint_index][keyframe_index].push_back(transformed_points[i].z-average.z);
            total_errors+=3;
        }
//        double sum_x_diff, sum_y_diff, sum_z_diff;
//        sum_x_diff=sum_y_diff=sum_z_diff=0;
//        for(size_t i=0;i<transformed_points.size();i++){
//            sum_x_diff+=std::abs(transformed_points[i].x-average.x);
//            sum_y_diff+=std::abs(transformed_points[i].y-average.y);
//            sum_z_diff+=std::abs(transformed_points[i].z-average.z);
//        }
//        error(error_index)=sum_x_diff;
//        error(error_index+1)=sum_y_diff;
//        error(error_index+2)=sum_z_diff;
//        error_index+=3;
        mappoint_index++;
    }
    unsigned int error_index=0;
    error.resize(total_errors);
    for(size_t i=0;i<mappoint_diffs.size();i++)
        for(size_t j=0;j<mappoint_diffs[i].size();j++)
            for(size_t k=0;k<mappoint_diffs[i][j].size();k++)
                error(error_index++)=mappoint_diffs[i][j][k];
}

void draw_keypoint_lines(vector<cv::Point3f> &points,vector<cv::Vec3b> &colors){
    for(auto &keyframe_keypoint_coords:mappoint_keyframe_keypoint_coords){
        cv::Point3f average(0,0,0);
        vector<cv::Point3f> global_coords;
        for(auto &keypoint_coords:keyframe_keypoint_coords){
            unsigned int keyframe_index=keypoint_coords.first;
            global_coords.push_back(Transform3d(keyframe_transforms[keyframe_index])(keypoint_coords.second));
            average+=global_coords.back();
        }
        average/=double(global_coords.size());
        for(auto &global_coord:global_coords)
            draw_3d_line_points(global_coord,average,100,cv::Vec3b(0,0,0),cv::Vec3b(255,0,0),points,colors);
    }
}

//struct Octree{
//    shared_ptr<Octree> subtrees[8];
//    cv::Point3f max;
//    cv::Point3f min;
//    cv::Point3f center;
//    vector<unsigned int> indices;
//    Octree(vector<cv::Point3f> &cloud, vector<unsigned int> &indcs){
//        indices=indcs;
//        center=cv::Point3f((min.x+max.x)/2,(min.y+max.y)/2,(min.z+max.z)/2);
//    }

//    for(size_t i=0;i<cloud.size();i++){
//        unsigned int index=0;
//        if(cloud[i].x>octree.center.x)
//            index += 1;
//        index <<= 1;
//        if(cloud[i].y>octree.center.y)
//            index += 1;
//        index <<= 1;
//        if(cloud[i].z>octree.center.z)
//            index += 1;
//        if(!octree.subtrees[index])
//            octree.subtrees[index]=shared_ptr<Octree>(new Octree());
//    }
//};

//Octree subsample_pointcloud(vector<cv::Point3f> cloud){
//    cv::Point3f min(numeric_limits<float>::lowest(),numeric_limits<float>::lowest(),numeric_limits<float>::lowest());
//    cv::Point3f max(numeric_limits<float>::max(),numeric_limits<float>::max(),numeric_limits<float>::max());
//    for(auto &point: cloud){//find the boundaries
//        min.x=std::min(min.x,point.x);
//        min.y=std::min(min.y,point.y);
//        min.z=std::min(min.z,point.z);
//        max.x=std::max(max.x,point.x);
//        max.y=std::max(max.y,point.y);
//        max.z=std::max(max.z,point.z);
//    }
//    //find the center point

//    return octree;
//}

void get_keyframe_transforms_and_3D_keypoint_coords(ucoslam::Map &m){
    for(ucoslam::MapPoint& mappoint:m.map_points) {
        map<unsigned int,cv::Point3f> keyframe_keypoint_coords;

        if(!mappoint.isBad){
            const auto &mappoint_keyframes=mappoint.getObservingFrames();
            map<unsigned int,cv::Point3f> keypoint_global_positions;
            for(auto &mappoint_keyframe: mappoint_keyframes){//go through the observing keypoints
                unsigned int keyframe_index=mappoint_keyframe.first;
                unsigned int keypoint_index=mappoint_keyframe.second;

                ucoslam::Frame &keyframe=m.keyframes[keyframe_index];
                cv::KeyPoint keypoint=keyframe.und_kpts[keypoint_index];
                //only use the keypoints in the first octave
                if(keyframe.depth[keypoint_index]<=2.0 && keypoint.octave==0 && !std::isnan(keypoint.pt.x) && !std::isnan(keypoint.pt.y) && !keyframe.isBad){
                    if(keyframe.depth[keypoint_index]>0){//check if the keypoint has a valid depth
                        keyframe_keypoint_coords[keyframe_index]=keyframe.get3dStereoPoint(keypoint_index);
                        if(keyframe_transforms.count(keyframe_index)==0){//save the transformation of the
                            keyframe.pose_f2g.inv().copyTo(keyframe_transforms[keyframe_index]);
                        }
                        keypoint_global_positions[keyframe_index]=Transform3d(keyframe_transforms[keyframe_index])(keyframe_keypoint_coords[keyframe_index]);
                    }
                }
            }

            if(keypoint_global_positions.size()>0){
                //get the average
                cv::Point3f average(0,0,0);
                for(auto &global_position : keypoint_global_positions)
                    average+=global_position.second;
                average/=double(keypoint_global_positions.size());

                //get the diffs
                vector<pair<unsigned int,double>> keyframe_diffs;
                for(auto &global_position : keypoint_global_positions)
                    keyframe_diffs.push_back(make_pair(global_position.first,cv::norm(global_position.second-average)));

                //get the mdedian

                std::sort(keyframe_diffs.begin(),keyframe_diffs.end(),[](pair<unsigned int,double> l,pair<unsigned int,double> r){
                   return l.second>r.second;
                });

                unsigned int median_keyframe_index1=keyframe_diffs[keyframe_diffs.size()/2].first;

                cv::Point3f median;


                if(keyframe_diffs.size()%2==1){
                    median=keypoint_global_positions[median_keyframe_index1];
                }
                else{
                    unsigned int median_keyframe_index2=keyframe_diffs[keyframe_diffs.size()/2+1].first;
                    median=(keypoint_global_positions[median_keyframe_index1]+keypoint_global_positions[median_keyframe_index2])/2.0;
                }

                //find the outliers and remove them
                for(auto &global_position : keypoint_global_positions){
                    if(cv::norm(median-global_position.second)>0.1){//the distance to median is more that 10 centimeters
                        unsigned int keyframe_index=global_position.first;
                        keyframe_keypoint_coords.erase(keyframe_index);
                    }
                }

            }

            if(keyframe_keypoint_coords.size()>0){
                mappoint_keyframe_keypoint_coords.push_back(keyframe_keypoint_coords);
                mappoint_coords.push_back(mappoint.getCoordinates());
            }
        }
    }
}

void register_keyframe_clouds(vector<depthmaps::EmbeddedRegistration> &ers){
    map<unsigned int, vector<cv::Vec3f>> keyframe_mappoint_coords;
    map<unsigned int, vector<cv::Vec3f>> keyframe_keypoint_global_coords;

    ers.resize(keyframe2frame_index.size());

    for(size_t i=0;i<mappoint_keyframe_keypoint_coords.size();i++){
        auto &keyframe_keypoint_coords=mappoint_keyframe_keypoint_coords[i];
        for(pair<unsigned int,cv::Point3f> p:keyframe_keypoint_coords){//move the keypoints from the keframe coordinates to the global coordinates
            unsigned int keyframe_index=p.first;
            cv::Point3f coords=Transform3d(keyframe_transforms[keyframe_index])(p.second);
            keyframe_keypoint_global_coords[keyframe_index].push_back(coords);
            keyframe_mappoint_coords[keyframe_index].push_back(mappoint_coords[i]);
        }
    }

    for(size_t i=0;i<keyframe2frame_index.size();i++){
        unsigned int keyframe_index=keyframe2frame_index[i].first;
        depthmaps::EmbeddedRegistration::Graph graph;
        graph.create(keyframe_mappoint_coords[keyframe_index],0,4);//create a graph using the mappoint coordinates
        ers[i].setParams(keyframe_keypoint_global_coords[keyframe_index],keyframe_mappoint_coords[keyframe_index],graph);//get the deformable registration for that keyframe
        ers[i].run();
    }
}

void apply_keyframe_deformations(vector<depthmaps::EmbeddedRegistration> &ers,const vector<vector<cv::Point3f>> &keyframe_clouds, vector<vector<cv::Point3f>> &keyframe_deformed_clouds){
    keyframe_deformed_clouds.resize(keyframe_clouds.size());
    for(size_t i=0;i<ers.size();i++){
        vector<cv::Vec3f> keyframe_cloud(keyframe_clouds.size()),keyframe_deformed_cloud(keyframe_clouds.size());
        for(size_t j=0;j<keyframe_clouds.size();j++)
            keyframe_cloud[j]=keyframe_clouds[i][j];

        ers[i].apply(keyframe_cloud,keyframe_deformed_cloud);

        for(size_t j=0;j<keyframe_clouds.size();j++)
            keyframe_deformed_clouds[i][j]=keyframe_deformed_cloud[j];
    }
}

void XYZImage_2_depthmap(const cv::Mat imgXYZ, const cv::Mat image,depthmaps::DepthMap &dm){
    if(dm.size!=imgXYZ.size)
        dm.create(imgXYZ.rows,imgXYZ.cols,true);
    for(int i=0;i<imgXYZ.rows;i++){
        depthmaps::DepthPixel *dm_row=dm.ptr<depthmaps::DepthPixel>(i);
        const cv::Vec3f *ixyz_row=imgXYZ.ptr<cv::Vec3f>(i);
        const cv::Vec3b *image_row=image.ptr<cv::Vec3b>(i);
        for(int j=0;j<imgXYZ.cols;j++)
            if(ixyz_row[j]!=cv::Vec3f(0,0,0)){
                dm_row[j]=ixyz_row[j];
                cv::Vec3b color=image_row[j];
                dm_row[j].setRGB(color[2],color[1],color[0]);
            }
    }
}

void read_keyframe_pointclouds(shared_ptr<RGBDReader> rgbd_reader, vector<vector<cv::Point3f>> &point_clouds, vector<vector<cv::Vec3b>> &pc_colors, vector<cv::Mat> &dms, double max_dist){
    point_clouds.resize(keyframe2frame_index.size());
    pc_colors.resize(keyframe2frame_index.size());
    dms.resize(keyframe2frame_index.size());
    for(size_t i=0;i<keyframe2frame_index.size();i++){
        unsigned int keyframe_index=keyframe2frame_index[i].first;
        unsigned int frame_index=keyframe2frame_index[i].second;
        rgbd_reader->readFrameAt(frame_index);
        rgbd_reader->getCurrCloud(point_clouds[i],pc_colors[i],keyframe_transforms[keyframe_index],max_dist);
        //cv::Mat XYZImage,image;
        //rgbd_reader->getCurrColor(image);
        rgbd_reader->getXYZImage(dms[i],keyframe_transforms[keyframe_index],max_dist);
        //XYZImage_2_depthmap(XYZImage,image,dms[i]);
    }
}

void transform_pointcloud(depthmaps::PointCloud &cloud,cv::Mat T){
    T.convertTo(T,CV_32FC1);
    for(int i=0;i<cloud.size();i++){
        cv::Mat v(4,1,CV_32FC1);

        cv::Vec3f vec=cloud[i].toVec3f();
        for(int j=0;j<3;j++)
            v.at<float>(j)=vec[j];

        v.at<float>(3)=1;
        cv::Mat transformed_point=T*v;
        cloud[i]=cv::Vec3f(transformed_point.rowRange(0,3));
    }
}

//bool find_plane(vector<cv::Point3f> point_cloud){
//    map<int,map<int,map<int,map<int,float>>>> scores;
//    for(int i=0;i<;i++){

//    }
//}

void print_usage() {
    cout<<"Args: <key_frames_map_file> <path_to_dataset> <dataset_type> [--orbslam2]"<<endl;
}

int main(int argc, char *argv[])
{
    if(argc<3) {
        print_usage();
        return -1;
    }
    bool orbslam2=false;

    if(argc>3)
        if(string(argv[3])=="--orbslam2")
            orbslam2=true;


    string map_path(argv[1]);
    string output_folder=map_path.substr(0,map_path.find_last_of('/')+1);

    shared_ptr<RGBDReader> rgbd_reader=RGBDReaderFactory::getReader(argv[3],argv[2]);

    map<unsigned long, unsigned long> frame2keyframe_index;
    map<unsigned int,map<unsigned int,cv::Point3f>> frame_mappoints;
    map<unsigned int,map<unsigned int,cv::Point3f>> frame_local_mappoints;
    map<unsigned long,cv::Mat> orbslam2_pose_converters;
    map<unsigned int,cv::Mat> initial_frame_transforms, optimized_frame_transforms;
    unsigned long last_frame_index;

    ucoslam::Map m;
    m.readFromFile(map_path);
    
    vector<vector<cv::Point3f>> keyframe_clouds,keyframe_deformed_clouds,gicp_clouds;
    vector<vector<cv::Vec3b>> keyframe_cloud_colors;
    vector<cv::Mat> keyframe_depthmaps, smooth_depthmaps;

    get_keyframe_transforms_and_3D_keypoint_coords(m);
    for(auto &keyframe:m.keyframes)
        if(keyframe_transforms.count(keyframe.idx)==1){//if the key frame has a transformation add the frame and keyframe indices to the maps
            frame2keyframe_index[keyframe.fseq_idx]=keyframe.idx;
            keyframe2frame_index.push_back(std::pair<unsigned int,unsigned int>(keyframe.idx,keyframe.fseq_idx));
        }

    read_keyframe_pointclouds(rgbd_reader, keyframe_clouds, keyframe_cloud_colors, keyframe_depthmaps, 1.5);//it also transforms the clouds according the map keyframe transforms

    //smooth the depthmaps
    smooth_depthmaps.resize(keyframe_depthmaps.size());
    for(size_t i=0;i<keyframe_depthmaps.size();i++){
        //depthmaps::PCLUtils::removePlane(keyframe_depthmaps[i],0.02);
        //depthmaps::DepthMapUtils::gaussianSmooth(keyframe_depthmaps[i],smooth_depthmaps[i],0.01);
    }

    vector<vector<cv::Point3f>> plane_removed_pointclouds(keyframe_depthmaps.size());
    vector<vector<cv::Vec3b>> plane_removed_colors(keyframe_depthmaps.size());

//    for(size_t i=0;i<keyframe_depthmaps.size();i++){
//        depthmaps::PointCloud pc(keyframe_depthmaps[i]);
//        for(depthmaps::DepthPixel &dp:pc){
//            plane_removed_pointclouds[i].push_back(dp.toPoint3f());
//            plane_removed_colors[i].push_back(dp.getRGB());
//        }
//    }

//    vector<depthmaps::PointCloud> plane_removed_pointclouds(keyframe_depthmaps.size());

    depthmaps::PointCloud reconstruction;
    depthmaps::Icp icp;
    cv::Mat T;

    for(size_t i=0;i<keyframe_clouds.size();i++)
        if(keyframe_clouds[i].size())
            write_pcd(output_folder+"/orig_cloud_"+to_string(i)+".pcd",keyframe_clouds[i],keyframe_cloud_colors[i]);

    Global_ICP gicp(keyframe_depthmaps,0.02);
//    gicp.execute(0.2,50);
//    gicp_clouds=gicp.getClouds();

//    for(size_t i=0;i<gicp_clouds.size();i++)
//        if(gicp_clouds[i].size())
//            write_pcd(output_folder+"/gicp_cloud_20cm"+to_string(i)+".pcd",gicp_clouds[i],keyframe_cloud_colors[i]);

    const int num_iterations=30;
    const pair<double,double> radius_interval(0.01,0.05);

    gicp.execute(radius_interval,num_iterations);

    gicp_clouds=gicp.getClouds(keyframe_clouds);

    for(size_t i=0;i<gicp_clouds.size();i++)
        if(gicp_clouds[i].size())
            write_pcd(output_folder+"/gicp_cloud_"+to_string(i)+".pcd",gicp_clouds[i],keyframe_cloud_colors[i]);

    vector<vector<cv::Mat>> transforms=gicp.getTransforms();

//    for(int i=0;i<transforms.size();i++){
//        cout<<"cloud num: "<<i<<endl;
//        if(i==10){
//            cv::Mat T=cv::Mat::eye(4,4,CV_64FC1);
//            for(int it=0;it<transforms[i].size();it++){
//                cout<<"iteraiton: "<<it<<endl;
//                cv::Mat transform;
//                transforms[i][it].convertTo(transform,CV_64FC1);
//                T=transform*T;
//                cout<<T<<endl;
//            }
//        }

//    }


    return 0;

//    for(size_t i=0;i<smooth_depthmaps.size();i++){
//        cout<<"Cloud number: "<<i<<endl;
//        depthmaps::PointCloud smooth_pointcloud(keyframe_depthmaps[i]);
//        if(reconstruction.size()>1 && smooth_pointcloud.size()>1){
//            //if(!T.empty())
//            //    transform_pointcloud(smooth_pointcloud,T);

//            icp.run(smooth_pointcloud,reconstruction,1,0.05,T);
//            transform_pointcloud(smooth_pointcloud,T);
//        }
//        //for(size_t j=0;j<transformed_pointcloud.size();j++)
//        reconstruction.insert(reconstruction.end(),smooth_pointcloud.begin(),smooth_pointcloud.end());
//    }

    reconstruction.saveToPCD(output_folder+"registered_icp_smooth.pcd");
//    vector<cv::Point3f> registered_clouds(reconstruction.size());

//    for(int i=0;i<reconstruction.size();i++)
//        registered_clouds[i]=reconstruction[i].toPoint3f();


//    vector<cv::Vec3b> registered_clouds_colors(reconstruction.size());
//    size_t index=0;
//    for(int i=0;i<keyframe_cloud_colors.size();i++)
//        for(int j=0;j<keyframe_cloud_colors[i].size();j++)
//            registered_clouds_colors[index++]=keyframe_cloud_colors[i][j];

//    write_pcd(output_folder+"registered_icp_smooth.pcd",registered_clouds,registered_clouds_colors);

    return 0;

    vector<cv::Point3f> points;
    vector<cv::Vec3b> colors;
    getTransformsPointCloud(frame2keyframe_index,rgbd_reader,points,colors,keyframe_transforms);
    draw_keypoint_lines(points,colors);
    write_pcd(output_folder+"map_transforms.pcd",points,colors);

//    SequenceOptimizer so(m);
//    so.getTransforms(initial_frame_transforms);
//    so.optimize();
//    so.getTransforms(optimized_frame_transforms);

//    for(auto transform:initial_frame_transforms){
//        cout<<"initial: "<<transform.second<<endl;
//        cout<<"optimized: "<<optimized_frame_transforms[transform.first]<<endl;
//        cout<<"----------"<<endl;
//    }

    unsigned int io_index=0;
    for(auto &keyframe_transform:keyframe_transforms){
        keyframe_io_index[keyframe_transform.first]=io_index;
        io_index+=6;
    }

    //optimization----------
    ucoslam::SparseLevMarq<double>::eVector io_vec;
    ucoslam::SparseLevMarq<double> solver;
    ucoslam::SparseLevMarq<double>::Params p;
    p.verbose=true;
    p.min_average_step_error_diff=0.0000001;
    solver.setParams(p);
    keyframe_transforms_to_io_vec(io_vec);
    solver.solve(io_vec,bind(error_function,placeholders::_1,placeholders::_2));
    io_vec_to_keyframe_transforms(io_vec);
    //optimization----------

    getTransformsPointCloud(frame2keyframe_index,rgbd_reader,points,colors,keyframe_transforms);
    draw_keypoint_lines(points,colors);
    write_pcd(output_folder+"keypoints_optimized.pcd",points,colors);


    for(auto &keyframe_transform:keyframe_transforms){
        unsigned int keyframe_index=keyframe_transform.first;
        cv::Mat transform=keyframe_transform.second;
        points.clear();
        colors.clear();
        unsigned int frame_index=m.keyframes[keyframe_index].fseq_idx;
        rgbd_reader->readFrameAt(frame_index);
        //rgbd_reader->getCurrCloud(points,colors,transform);
        cv::Mat xyz_image,color;
        rgbd_reader->getXYZImage(xyz_image,transform);
        rgbd_reader->getCurrColor(color);
        char frame_num[10];
        std::sprintf(frame_num,"%04d",keyframe_index);
        write_pcd("/media/hamid/Data/frames/"+string(frame_num)+".pcd",xyz_image,color);
    }

    return 0;



//    unsigned int io_index;
//    for(auto &transfrom:initial_frame_transforms){
//        cv::Mat r,t;
//        cv::Rodrigues(transform.second(cv::Range(0,3),cv::Range(0,3)),r);
//        for(int i=0;i<3:i++)

//    }

    //solver.solve(io_vec,bind(&error_function,placeholders::_1,placeholders::_2));
    //retrieve the depth values for keypoints in each frame
//    cv::Mat depth,color,xyz;
//    for(auto &frame2keyframe: frame2keyframe_index){

//        unsigned int frame_index=frame2keyframe.first;
//        unsigned int keyframe_index=frame2keyframe.second;

//        cout<<frame_index<<endl;

//        rgbd_reader->getFrameAt(frame_num,depth,color);
//        rgbd_reader->getCurrCloud(points,colors,pose_converter);

//        vector<cv::KeyPoint> &keypoints=m.keyframes[keyframe_index].und_kpts;//get the undistorted keypoints for that keyframe

////        for(auto &p:frame_mappoints[frame_index]) { //go through the map points in that key frame
////            auto keypoint_index=p.first;//get the index of the kepoint related to the map point
////            cv::KeyPoint &keypoint=keypoints[keypoint_index];//retrieve the keypoint
////            if(std::isnan(keypoint.pt.x) || std::isnan(keypoint.pt.y))//if it does not have valid coordinates in the image continue
////                continue;
////            int row=keypoint.pt.y+0.5;
////            int col=keypoint.pt.x+0.5;
////            cv::Vec3d coords=xyz.at<cv::Vec3d>(row,col);
////            frame_local_mappoints[frame_index][keypoint_index]=cv::Vec3f(coords);
////        }
//    }

//    if(orbslam2) {
//        read_orbslam2_keyframes(argv[1],frame2keyframe_index,orbslam2_pose_converters,last_frame_index);
//    }

//    cv::Mat dist_coeffs();

//    vector<cv::Point3d> points;
//    vector<cv::Vec3b> colors;



//    int num_used_frames=0;
//    for(unsigned long frame_index=0; rgbd_reader->getNextFrame(depth,color); frame_index++) {

//        if(frame2keyframe_index.count(frame_index)==0)
//            continue;
//        num_used_frames++;
//        cout<<frame_index<<endl;

//        unsigned long keyframe_index=frame2keyframe_index[frame_index];

//        cv::Mat pose_converter;

//        if(orbslam2) {
//            pose_converter=orbslam2_pose_converters[frame_index];
//        }
//        else {
////            vector<cv::KeyPoint> &keypoints=m.keyframes[keyframe_index].und_kpts;

////            vector<float> &depths=m.keyframes[keyframe_index].depth;
//            pose_converter=m.keyframes[keyframe_index].pose_f2g.inv();
//        }
////            for(auto &p:frame_mappoints[frame_index]){
////                auto keypoint_index=p.first;
////                cv::KeyPoint &keypoint=keypoints[keypoint_index];
////                if(std::isnan(keypoint.pt.x) || std::isnan(keypoint.pt.y))
////                    continue;
////                cv::Vec3d mappoint_coords=p.second;
////                cv::Vec3d keypoint_coords(keypoint.pt.x,keypoint.pt.y,1);
////                double z=undistorted_d.at<char16_t>(keypoint.pt.y,keypoint.pt.x)/1000.0;
////                if(z==0 || z>2)
////                    continue;
////                keypoint_coords=cam_mat.inv()*keypoint_coords*z;
////                auto tmp_coords=cv::Matx44f(pose_converter)*cv::Vec4f(keypoint_coords[0],keypoint_coords[1],keypoint_coords[2],1);
////                for(int i=0;i<3;i++)
////                    keypoint_coords[i]=tmp_coords[i];
////                draw_3d_line_points(keypoint_coords,mappoint_coords,100,cv::Vec3b(255,0,0),points,colors);
////            }
////        }

//        rgbd_reader->getCurrCloud(points,colors,pose_converter);
//        cv::imshow("BGR",color);
//        cv::waitKey(10);

//        if(frame_index==last_frame_index)
//            break;
//    }



    //write_pcd("out.pcd",points,colors);
//    cout<<f.getMapPoints().size()<<endl;
//    cout<<f.getCameraCenter()<<endl;
//    cout<<f.getCameraDirection()<<endl;


    return 0;
}
