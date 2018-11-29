
#include "mapviewer.h"
#include <opencv2/highgui/highgui.hpp>


int main( int  argc , char**  argv )
{
    if (argc<2){cerr<<"Usage: world [-system]"<<endl;return -1;}
    std::shared_ptr<ucoslam::Map> map=std::make_shared<ucoslam::Map>();
    bool isSlam=false;
    if(argc>=3){
        if(string(argv[2])=="-system")
            isSlam=true;
    }
    if(isSlam){
        ucoslam::UcoSlam SlamSystem;
        SlamSystem.readFromFile(argv[1]);
        map=SlamSystem.getMap();
    }
    else
        map->readFromFile(argv [1]);


    cout<<"Npoints="<<map->map_points.size()<<endl;
    cout<<"NFrames="<<map->keyframes.size()<<endl;
    ucoslam::MapViewer Theviewer;
    Theviewer.set("mode","0");
    bool finish=false;

    while(!finish){
         int k=Theviewer.show( map ) ;
        if (k==27)finish=true;
    }
    return 0;

}
