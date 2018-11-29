#include "frameextractor.h"
#include <thread>
#include <opencv2/imgproc.hpp>
#include "stuff/timers.h"
#include "stuff/utils.h"
#include "optimization/ippe.h"
namespace ucoslam {


void FrameExtractor::toStream(std::ostream &str)const
{
    uint64_t sig=1923123;
    str.write((char*)&sig,sizeof(sig));    
    str.write((char*)&_counter,sizeof(_counter));
    str.write((char*)&_removeFromMarkers,sizeof(_removeFromMarkers));
    str.write((char*)&_detectMarkers,sizeof(_detectMarkers));
    str.write((char*)&_detectKeyPoints,sizeof(_detectKeyPoints));
     str.write((char*)&_markerSize,sizeof(_markerSize));
     _mdetector.toStream(str);
}

void FrameExtractor::fromStream(std::istream &str){
    uint64_t sig=1923123;
    str.read((char*)&sig,sizeof(sig));
    if (sig!=1923123) throw std::runtime_error("FrameExtractor::fromStream invalid signature");
     str.read((char*)&_counter,sizeof(_counter));
    str.read((char*)&_removeFromMarkers,sizeof(_removeFromMarkers));
    str.read((char*)&_detectMarkers,sizeof(_detectMarkers));
    str.read((char*)&_detectKeyPoints,sizeof(_detectKeyPoints));
     str.read((char*)&_markerSize,sizeof(_markerSize));
     _mdetector.fromStream(str);

}

FrameExtractor::FrameExtractor(){}


void FrameExtractor::setParams(const Params &params){
     _mdetector.setParameters(params.aruco_DetectorParams);
    _markerSize=params.aruco_markerSize;
}

void FrameExtractor::process_rgbd(const cv::Mat &image, const cv::Mat &depthImage,const ImageParams &ip,Frame &frame, uint32_t frameseq_idx , uint32_t fidx )throw (std::exception){
    assert(ip.bl>0);
    assert(depthImage.type()==CV_16UC1);
         process(image,ip,frame,frameseq_idx,fidx);
        //  float fb=ip.bl*ip.fx();
         //now, add the extra info to the points
         for(size_t i=0;i<frame.kpts.size();i++){
             //convert depth
             if (depthImage.at<uint16_t>(frame.kpts[i].pt)!=0){
                 frame.depth[i]=depthImage.at<uint16_t>(frame.kpts[i].pt)*ip.rgb_depthscale;
                // frame.mvuRight[i]=frame.kpts[i].pt.x- fb/frame.depth[i];//find projection in IR camera
             }
         }
}
struct marker_analyzer{

    marker_analyzer(vector<cv::Point2f> &m){
        bax = m[1].x - m[0].x;
        bay = m[1].y - m[0].y;
        dax = m[2].x - m[0].x;
        day =  m[2].y - m[0].y;
        a=m[0];b=m[1];d=m[2];

    }

    bool isInto(const cv::Point2f &p)const{
        if ((p.x - a.x) * bax + (p.y - a.y) * bay < 0.0) return false;
        if ((p.x - b.x) * bax + (p.y - b.y) * bay > 0.0) return false;
        if ((p.x - a.x) * dax + (p.y - a.y) * day < 0.0) return false;
        if ((p.x - d.x) * dax + (p.y - d.y) * day > 0.0) return false;

        return true;
    }
    float bax, bay , dax  ,day;
    cv::Point2f a,b,d;

};
void FrameExtractor::process(const cv::Mat &image,const ImageParams &ip,Frame &frame,uint32_t frameseq_idx,uint32_t fidx)throw (std::exception){
    assert(image.size()==ip.CamSize);

    frame.clear();
    ScopedTimerEvents tem("FrameExtractor::process");
     if (image.channels()==3)
        cv::cvtColor(image,_imgrey,CV_BGR2GRAY);
    else _imgrey=image;





    std::thread aruco_thread( [&]{
        if (_detectMarkers){
            _mdetector.detect(_imgrey,frame.markers);
            //remove elements from the black list
//#warning "remove this"
//            std::vector<int> black_list
//                    ={0,10,300,330,350};
//            frame.markers.erase( std::remove_if(frame.markers.begin(),frame.markers.end(),[black_list](const aruco::Marker &m){ return std::find(black_list.begin(),black_list.end(), m.id)!=black_list.end();}),frame.markers.end());
//            //now, apply IPPE to detect the locations
            for(const auto&m:frame.markers){
                auto sols=IPPE::solvePnP_(_markerSize,m,ip.CameraMatrix,ip.Distorsion);
                MarkerPosesIPPE mp;
                mp.errs[0]=sols[0].second;
                mp.errs[1]=sols[1].second;
                mp.sols[0]=sols[0].first;
                mp.sols[1]=sols[1].first;
                mp.err_ratio=sols[1].second/sols[0].second;
                frame.markers_solutions.push_back(mp);
            }
            for(auto&m:frame.markers)
                m.ssize=_markerSize;

        }
    }
    );
     aruco_thread.join();

    if (debug::Debug::getLevel()>=10)
        image.copyTo(frame.image);

    //create a reduced image version for fern database
    //use the scale factor making the image width of 128 pix
    float sc=128./float(image.cols);

    cv::Mat aux;
    cv::resize(image,aux,cv::Size( float(image.cols)*sc,float(image.rows)*sc));
    if (aux.type()!=CV_8UC1)
        cv::cvtColor(aux,frame.smallImage,CV_BGR2GRAY);
    else frame.smallImage=aux;
    //remove keypoints into markers??

    tem.add("Keypoint/Frames detection");

    if (_removeFromMarkers  ){
        vector<marker_analyzer> manalyzers;
        for(auto m:frame.markers)
            manalyzers.push_back(marker_analyzer(m));
        vector<cv::KeyPoint> kp2;
        kp2.reserve(frame.kpts.size());
        cv::Mat desc2(frame.desc.size(),frame.desc.type());
        for(size_t i=0;i<frame.kpts.size();i++){
            bool isIntoAny=false;
            for(uint a=0;a<manalyzers.size() && !isIntoAny;a++)
                if (manalyzers[a].isInto(frame.kpts[i].pt)) isIntoAny=true;
            if (!isIntoAny){
                frame.desc.rowRange(i,i+1).copyTo(desc2.rowRange(kp2.size(),kp2.size()+1));
                kp2.push_back(frame.kpts[i]);
            }
        }
        desc2.resize(kp2.size() ,desc2.cols);
        frame.desc=desc2;
        frame.kpts=kp2;
    }


    tem.add("remove from markers");

    //remove distortion

    if (frame.kpts.size()>0){
        vector<cv::Point2f> pin;pin.reserve(frame.kpts.size());
        for(auto p:frame.kpts) pin.push_back(p.pt);
        undistortPoints(pin,ip );

        frame.und_kpts=frame.kpts;
        for ( size_t i=0; i<frame.kpts.size(); i++ )
            frame.und_kpts[i].pt=pin[i];
    }

    tem.add("undistort");
    //remove distortion of the marker points if any
    frame.und_markers=frame.markers;
    for(auto &m:frame.und_markers)
        undistortPoints(m,ip);


    frame.nonMaxima.resize(frame.und_kpts.size());
    for(auto &v:frame.nonMaxima) v=false;

    frame.ids.resize(frame.und_kpts.size());
    //set the keypoint ids vector to invalid
    uint32_t mval=std::numeric_limits<uint32_t>::max();
    for(auto &ids:frame.ids) ids=mval;
    //create the grid for fast access


    //set the frame id
  // assert (fidx!=std::numeric_limits<uint32_t>::max()) ;
    frame.idx=fidx;
    frame.fseq_idx=frameseq_idx;
    frame.imageParams=ip;


    frame.depth.resize(frame.und_kpts.size());
    for(size_t i=0;i<frame.depth.size();i++) frame.depth[i]=0;
    frame.create_kdtree();//last thing
 }
}
