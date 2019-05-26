#include "fractallabelers/fractalmarkerset.h"
#include "dictionary.h"
#include <cstdio>
#include <opencv2/highgui/highgui.hpp>
#include <string>

using namespace std;

class CmdLineParser
{
    int argc;char** argv;public:
    CmdLineParser(int _argc, char** _argv): argc(_argc), argv(_argv){}
    bool operator[](string param)
    {int idx = -1;   for (int i = 0; i < argc && idx == -1; i++)if (string(argv[i]) == param) idx = i;return (idx != -1); }
    string operator()(string param, string defvalue = "-1"){int idx = -1;for (int i = 0; i < argc && idx == -1; i++)if (string(argv[i]) == param) idx = i;if (idx == -1) return defvalue;else return (argv[idx + 1]);}
};

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    try
    {
        CmdLineParser cml(argc, argv);
        if (argc < 3 || cml["-h"])
        {
            cerr << "Usage: fractal_config.yml <nlevels>:number of levels "
                    "<nbits>:number of bits for the last level " //n(fm) in paper
                    "<pixratio>:Pixel size ratio (0-1) "
                    "<ninfobits>:number of bit for info in each level "
                    "<inforatio>:Ratio bits info "
                    "[-s pixSize: last level (1px default)]" << endl;

            cerr << endl;
            return -1;
        }

        //Number of levels
        int nLevels = stoi(argv[2]);

        //NBITS Last level, n(fm) in paper
        int nBits = stoi(argv[3]);

        //PIXRATIO fn(bit size) = pixRatio * k(fn)
        //For ex. pixRatio = 0.5, then 1 bit of fn marker is the half size of the total length marker fn+1 (with border)
        //For ex. pixRatio = 1, then 1 bit of fn marker is the same size of the total length marker fn+1 (with border)
        float pixRatio = stof(argv[4]);

        //Tamaño de la banda gris en cada nivel (valores 1..x)
        int ninfobits = stoi(argv[5]);

        //Ratio de aumento de la banda gris conforme nos alejamos del marcador mas interno (valores 1..x)
        int infoRatio = stoi(argv[6]);

        //PIXSize Last Level
        int pixSize = stoi(cml("-s", "1"));

        aruco::FractalMarkerSet fractalmarkerset;
        fractalmarkerset.create(nLevels, nBits, pixRatio, ninfobits, infoRatio, pixSize);
        //Save configuration file
        cv::FileStorage fs(argv[1], cv::FileStorage::WRITE);
        fractalmarkerset.saveToFile(fs);
    }
    catch (std::exception& ex)
    {
        cout << ex.what() << endl;
    }
}
