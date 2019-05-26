#include "global_icp.h"
#include "basictypes/misc.h"
#include <set>

using namespace std;

Global_ICP::Global_ICP(const std::vector<cv::Mat>  &in_depthmaps, float subsample_radius)
{
    clouds.resize(in_depthmaps.size());
//#pragma omp parallel for
    for(size_t i=0;i<in_depthmaps.size();i++){
        cv::Mat depthmap;
        in_depthmaps[i].copyTo(depthmap);
        if(depthmap.depth() != CV_32F && depthmap.depth() != CV_64F)
            throw runtime_error("The depthmap matrices need to be of a floating point format.");
        if(depthmap.channels() < 3 || depthmap.channels() > 4)
            throw runtime_error("The dephtmap matrices either 3 or 4 channels.");
        if(depthmap.depth() == CV_32F){
            if(depthmap.channels() == 3)
                subsampleDepthmap2Pointcloud<cv::Vec3f>(depthmap,clouds[i],subsample_radius);
            else
                subsampleDepthmap2Pointcloud<cv::Vec4f>(depthmap,clouds[i],subsample_radius);
        }
        if(depthmap.depth() == CV_64F){
            if(depthmap.channels() == 3)
                subsampleDepthmap2Pointcloud<cv::Vec3d>(depthmap,clouds[i],subsample_radius);
            else
                subsampleDepthmap2Pointcloud<cv::Vec4d>(depthmap,clouds[i],subsample_radius);
        }
    }
    initialize(subsample_radius);
}

template<typename T>
double distance(T &val1, T &val2){
    double diff_x=val1[0]-val2[0];
    double diff_y=val1[1]-val2[1];
    double diff_z=val1[2]-val2[2];
    return std::sqrt(diff_x*diff_x+diff_y*diff_y+diff_z*diff_z);
}

template<typename T>
void Global_ICP::subsampleDepthmap2Pointcloud(cv::Mat &depthmap,std::vector<cv::Point3f> &cloud, float subsample_radius){
    //create arrays for fast access
    vector<T*> fast_access_mat(depthmap.rows);
    for(int r=0;r<depthmap.rows;r++)
        fast_access_mat[r]=depthmap.ptr<T>(r);
    //extract points to put in the point cloud
    for(int r=0;r<depthmap.rows;r++){
        bool is_last_row = (r==depthmap.rows-1);
        for(int c=0;c<depthmap.cols;c++){
            T &val=fast_access_mat[r][c];
            //ignore the pixel if it is invalid
            if(std::isnan(val[0]))
                continue;
            if(depthmap.channels()>3){
                int64_t last_channel=0;
                if(depthmap.depth() == CV_32F)
                    last_channel=*((int32_t*)&val[3]);
                else if(depthmap.depth() == CV_64F)
                    last_channel=*((int64_t*)&val[3]);
                if((last_channel & 1) == 0)
                    continue;
            }
            //add the pixel to the point cloud
            cloud.push_back(cv::Point3f(val[0],val[1],val[2]));
            //invalidate neighbouring pixels if they are too close
            if(c!=depthmap.cols-1){
                if(distance(val,fast_access_mat[r][c+1])<subsample_radius)
                    fast_access_mat[r][c+1][0]=NAN;
                if(!is_last_row)
                    if(distance(val,fast_access_mat[r+1][c+1])<subsample_radius)
                        fast_access_mat[r+1][c+1][0]=NAN;
            }
            if(!is_last_row){
                if(distance(val,fast_access_mat[r+1][c])<subsample_radius)
                    fast_access_mat[r+1][c][0]=NAN;
                if(c!=0)
                    if(distance(val,fast_access_mat[r+1][c-1])<subsample_radius)
                        fast_access_mat[r+1][c-1][0]=NAN;
            }
        }
    }
}

Global_ICP::Global_ICP(const std::vector<std::vector<cv::Point3f> > &in_clouds, float subsample_radius)
{
    clouds=in_clouds;
    initialize(subsample_radius);
}

void Global_ICP::initialize(float subsample_radius){
    num_clouds=clouds.size();

    trees.resize(num_clouds);
    transforms.resize(num_clouds);
    cout<<"subsampling the clouds...";

    random_device rand_d;
    vector<unsigned int> seeds(num_clouds);
    for(size_t i=0;i<num_clouds;i++)
        seeds[i]=rand_d();
#pragma omp parallel for
    for(size_t i=0;i<num_clouds;i++)
        clouds[i]=subsamplePointcloud(clouds[i],subsample_radius,seeds[i]);
    cout<<"done!"<<endl;
}

void Global_ICP::execute(pair<double,double> radius_range, int num_iterations){
    std::vector<std::vector<cv::Point3f>> curr_clouds=clouds;

    for(size_t c=0;c<transforms.size();c++)
        transforms[c].resize(num_iterations);

    double radius_interval_size=radius_range.second-radius_range.first;
    double max_distance=radius_range.second;

    for(int i=0;i<num_iterations;i++){
        cout<<"gicp iteration: "<<i<<endl;
        cout<<"Building the KDtrees.."<<endl;

#pragma omp parallel for
        for(size_t i=0;i<num_clouds;i++)
            trees[i].build(curr_clouds[i]);
        cout<<"Done!"<<endl;

        double radius=(radius_interval_size*(num_iterations-i))/num_iterations+radius_range.first;
        cout<<radius<<endl;
#pragma omp parallel for
        for(size_t c=0;c<curr_clouds.size();c++){//loop over clouds to find the transforms
            transforms[c][i]=cv::Mat::eye(4,4,CV_32FC1);
            if(curr_clouds[c].empty())
                continue;
            vector<cv::Point3f> source,target;
            for(size_t p=0;p<curr_clouds[c].size();p++){//loop over points
                vector<pair<cv::Point3f,double>> closest_points;
                for(size_t t=0;t<trees.size();t++){//loop over trees
                    if(curr_clouds[t].empty() || t==c)
                        continue;

                    vector<pair<uint32_t,double>> result=trees[t].radiusSearch(curr_clouds[t],curr_clouds[c][p],radius);
                    if(result.size()>0){
                        double distance=result[0].second;
                        size_t index=result[0].first;;
                        //int im=importance(distance,max_distance,i,num_iterations);
                        pair<cv::Point3f,double> closest_point(curr_clouds[t][index],distance);
                        //cout<<im<<endl;
                        //for(int k=0;k<im;k++)
                            closest_points.push_back(closest_point);
//                        if(c==10 && i==77)
//                            cout<<result[0].second<<endl;
                    }
                }
                if(!closest_points.empty()){
                    bool take_average=false;
                    source.push_back(curr_clouds[c][p]);
                    if(take_average){
                        cv::Point3f average(0,0,0);
                        for(pair<cv::Point3f,double> &cp:closest_points)
                            average += cp.first;
                        average /= double(closest_points.size());
                        target.push_back(average);
                    }
                    else{//get the closest point in any cloud
                        double best_dist=numeric_limits<double>::max();
                        cv::Point3f best_point;
                        for(pair<cv::Point3f,double> &cp:closest_points){
                            if(best_dist>cp.second){
                                best_point=cp.first;
                                best_dist=cp.second;
                            }
                        }
                        target.push_back(best_point);
                    }
                }
            }
            if(source.size()>0)
                transforms[c][i]=ucoslam::rigidBodyTransformation_Horn1987(source,target,true);
        }
        for(size_t c=0;c<curr_clouds.size();c++){//loop over clouds to apply the transforms
            if(curr_clouds[c].empty())
                continue;
            transformPointcloud(curr_clouds[c],transforms[c][i]);
        }
    }
}

void Global_ICP::transformPointcloud(vector<cv::Point3f> &cloud, cv::Mat T){
    T.convertTo(T,CV_32FC1);
    cv::Matx44f trasform(T);
    cv::Vec4f mat_p;
    for(cv::Point3f &p:cloud){
        mat_p[0]=p.x;
        mat_p[1]=p.y;
        mat_p[2]=p.z;
        mat_p[3]=1;

        mat_p=trasform*mat_p;

        p.x=mat_p[0];
        p.y=mat_p[1];
        p.z=mat_p[2];
    }

}

vector<vector<cv::Point3f>> Global_ICP::getClouds(const std::vector<std::vector<cv::Point3f> > &in_clouds, int num_iterations){
    vector<vector<cv::Point3f>> transformed_clouds=in_clouds;
    if(num_iterations==-1 && transforms.size()>0)
        num_iterations=transforms[0].size();
    cout<<"Calculating the final transforms and transforming the clouds...";
#pragma omp parallel for
    for(size_t c=0;c<transforms.size();c++){
        cv::Mat transform=cv::Mat::eye(4,4,CV_64FC1);
        //get the final transform
        for(size_t i=0;i<num_iterations;i++){
            cv::Mat T;
            transforms[c][i].convertTo(T,CV_64FC1);
            transform=T*transform;
        }
        //transform the cloud
        transformPointcloud(transformed_clouds[c],transform);
    }
    cout<<"Done."<<endl;
    return transformed_clouds;
}

vector<vector<cv::Mat>> Global_ICP::getTransforms(){
    return transforms;
}

double importance_func(double input, double mu, double max_dist){
    return std::exp(-(input-mu)*(input-mu)/((max_dist/2)*(max_dist/2)));

}

int Global_ICP::importance(double distance,double max_distance,int iteration,int num_iterations){
    double mu=(max_distance*(num_iterations-1-iteration))/(num_iterations-1);
    double min_value;
    if(mu<max_distance/2)
        min_value=importance_func(max_distance,mu,max_distance);
    else
        min_value=importance_func(0,mu,max_distance);
    double value=importance_func(distance,mu,max_distance);
    return std::round(value/min_value);
}

vector<cv::Point3f> Global_ICP::subsamplePointcloud(const vector<cv::Point3f> &in_cloud, float radius, unsigned int seed){
    vector<cv::Point3f> downsampled;

    picoflann::KdTreeIndex<3,picoflann_point3f_adaptor> tree;
    tree.build(in_cloud);

    set<size_t> choosables;
    for(size_t i=0;i<in_cloud.size();i++)
        choosables.insert(i);

    uniform_int_distribution<size_t> int_dist(0,in_cloud.size()-1);
    mt19937_64 rand_generator(seed);

    while(!choosables.empty()){

        size_t index=int_dist(rand_generator);
        if(choosables.count(index)>0){
            downsampled.push_back(in_cloud[index]);
            vector<pair<uint32_t,double>> points=tree.radiusSearch(in_cloud,in_cloud[index],radius);
            for(size_t i=0;i<points.size();i++){
                if(choosables.count(points[i].first))
                    choosables.erase(choosables.find(points[i].first));
            }
        }

    }

    return downsampled;

}
