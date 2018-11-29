#include <ucoslamtypes.h>
#include <iostream>
using namespace  std;

int main(int argc,char **argv){
	
    if (argc!=2){cerr<<"Usage: fileout.yml"<<endl;return -1;}
    ucoslam::Params params;
    params.saveToYMLFile(argv[1]);
 }
