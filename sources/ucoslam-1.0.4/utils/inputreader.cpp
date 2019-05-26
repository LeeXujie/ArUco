#include "inputreader.h"
#include "dirreader.h"
#include <opencv2/highgui/highgui.hpp>
InputReader::InputReader( )
{
}
void InputReader::open(std::string path){
    vcap.open(path);
    if(!vcap.isOpened()){//try dir
        files=DirReader::read(path,"",DirReader::Params(true));
        imgIdx=0;
    }
    isLive=false;
}
void InputReader::open(int cameraIdx){
    vcap.open(cameraIdx);
    isLive=true;
    imgIdx=0;
}

bool InputReader::isOpened()const{
    if(vcap.isOpened())return true;
    if(files.size()!=0)return true;
    return false;
}
void InputReader:: set(int v,double val){
    if( vcap.isOpened()){
            vcap.set(v,val);
        }
    else{//images
        if(v==CV_CAP_PROP_POS_FRAMES && val<files.size()) imgIdx=val;
    }
}
double InputReader:: get(int v){

    if( vcap.isOpened()){
        if( isLive && v==CV_CAP_PROP_POS_FRAMES){
            return imgIdx;
        }

             else return vcap.get(CV_CAP_PROP_POS_FRAMES);
        }
    else{//images
        if(v==CV_CAP_PROP_POS_FRAMES) return imgIdx;
    }

    throw std::runtime_error("InputReader::get Should not get here");
}
int InputReader::getCurrentFrameIndex(  ){

    if(!isOpened()) return -1;
    if(vcap.isOpened()){
        if (isLive) return imgIdx;
        else return  int(vcap.get(CV_CAP_PROP_POS_FRAMES));
    }
    else{
        return imgIdx;
    }
}

InputReader & 	InputReader::operator>> (cv::Mat &image){
    if ( grab())
        retrieve(image);
    else image=cv::Mat();
    return *this;
}

bool InputReader::grab(){
    if(vcap.isOpened()){
        if(isLive)imgIdx++;
        return vcap.grab();
    }

    do{
     fileImage=cv::imread(files[imgIdx++]);
    }while(fileImage.empty() &&imgIdx<files.size());

    return !fileImage.empty();
}

void InputReader::retrieve(cv::Mat &im){
    if(vcap.isOpened()){
        vcap.retrieve(im);
    }
    else fileImage.copyTo(im);

}
