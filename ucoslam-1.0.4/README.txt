UcoSLAM
====

UcoSLAM is a library for Simultaneous Localization and Mapping using keypoints that able to operate with
monocular cameras, stereo cameras, rgbd cameras. Additionally, our library is fully integrated with the ArUco library
for detecting squared fiducial markers. They can be placed in the environment to improve tracking.


##
## Main feafures:

UcoSLAM is a complete reimplementation of ORBSLAM2, with additional features such as:

	* Ability to save/load maps generated
	* Ability to use  markers to enhance, initialization, tracking and long-term relocalization. In addition, markers allows to estimate the real map scale from monocular cameras.  
		It is integrated with the ArUco library.
	* Only one external dependency, OpenCV. The rest of required packages are into the library. Compilation is extremmely easy.
	* Ability to choose any type of keypoint, not restricted to ORB. 
	* Multiplatform, ready to be compiled in Windows and Linux systems
	* A easy-to-use graphical user interface to process your videos, save and visualize your maps, calibrating your camera, etc.
	* PPA repository for Ubuntus and ready-to-use packages for Windows. You do not need to be a developer to use it.
	

## 
## Download:

	The source and precompiled binaries for Windows can be downloaded in https://sourceforge.net/projects/ucoslam/.

	For Ubuntu users, you can add the ppa repository https://launchpad.net/~rmsalinas/+archive/ubuntu/ucoslam/ which has the library, and also the 
	graphical user interface program UcoSLAM_GUI
		
## 
## Build from source

    in Ubuntu >= 16.04
        sudo apt-get install cmake libopencv-dev qtbase5-dev libqt5opengl-dev ubuntu-dev-tools

    download .zip file from https://sourceforge.net/projects/ucoslam/ and uncompress. Then, do
    cd dir_uncompressed
    mkdir build
    cd build
    cmake ../ -DBUILD_GUI=ON
    make -j4
    [sudo make install ] //will install it on your system
    
    
## 
## Test data

As part of our work, we have generated a repository with the most famous benchmarks for SLAM such as Kitti, Eroc-MAV, TUM, etc. The repository contains all the datasets 
with groundtruth using the same naming convention. It makes extremmely easy to compare the performance of different SLAM methods. 
The repository is publicly available at : https://mega.nz/#F!YsU2AY7L!Of0oChqpFBh34Y0-GOQ7VQ

    
## 
## Licensing

UcoSLAM is released under GPLv3 license (see License-gpl.txt).
Please see Dependencies.md for a list of all included code and library dependencies which are not property of the authors of UcoSLAM.

For a closed-source version of UcoSLAM for commercial purposes, please contact the authors.

If you use UcoSLAM in an academic work, please cite the most relevant publication associated by visiting:
http://www.uco.es/investiga/grupos/ava/node/62

In order to encourage private companies to ask for the commercial license, part of has been obfuscated and by default,
the open version of the library has some minor limitations that prevents its use in commercial products. In particular, the maximum allowed resolution 
is limited, and the maximum number of frames that can be processed without reinitialization is limited to approx 25 minutes of continuos operation.

 

 

