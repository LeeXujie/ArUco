#include "ucoslam.h"
#include "map.h"
#include "mapviewer.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "basictypes/debug.h"

class CmdLineParser{int argc; char **argv;
                public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue=""){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }
                    std::vector<std::string> getAllInstances(string str){
                        std::vector<std::string> ret;
                        for(int i=0;i<argc-1;i++){
                            if (string(argv[i])==str)
                                ret.push_back(argv[i+1]);
                        }
                        return ret;
                    }
                   };

int main(int argc, char* argv[]){
    CmdLineParser cml(argc,argv);
    if(argc<4 || cml["-h"]){
        cout<<"Usage: <video1> <video2> <stereo_calibration_file> [-voc path] [-params ucoslamparams.yml]"<<endl;
        return -1;
    }
    ucoslam::ImageParams image_params;
    image_params.readFromXMLFile(argv[3]);
    cv::Mat rectified_image[2];
    cv::VideoCapture video[2];
    for(int i=0;i<2;i++){
        video[i].open(argv[i+1]);
        if(!video[i].isOpened())
            throw runtime_error(string("Cannot open video file at:")+argv[i+1]);
    }

    ucoslam::Params sparams;
    sparams.detectMarkers=false;
    sparams.KFMinConfidence=0.8;
    sparams.runSequential=false;

    if(cml["-params"])
        sparams.readFromYMLFile(cml("-params"));

    auto smap=make_shared<ucoslam::Map>();
    ucoslam::UcoSlam system;
    ucoslam::MapViewer mv;
    system.setParams(smap,sparams,cml("-voc",""));
    char key=0;
    while( video[0].grab() && video[1].grab() && key!=27){
        int frameNumber=video[0].get(CV_CAP_PROP_POS_FRAMES);
        video[0].retrieve(rectified_image[0]);
        video[1].retrieve(rectified_image[1]);
        cv::Mat pose=system.processStereo(rectified_image[0],rectified_image[1],image_params,frameNumber );
        key=mv.show(smap,rectified_image[0],pose,"");
    }
    smap->saveToFile("world.map");

    return 0;
}
