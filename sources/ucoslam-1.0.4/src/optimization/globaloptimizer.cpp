/**
* This file is part of  UCOSLAM
*
* Copyright (C) 2018 Rafael Munoz Salinas <rmsalinas at uco dot es> (University of Cordoba)
*
* UCOSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* UCOSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with UCOSLAM. If not, see <http://wwmap->gnu.org/licenses/>.
*/
#include "globaloptimizer.h"
 #include "globaloptimizer_g2o.h"
 //#ifdef USE_CVBA
//#include "globaloptimizer_cvba_pba.h"
//#include "globaloptimizer_cvba_ucosba.h"
//#endif
namespace ucoslam{
std::shared_ptr<GlobalOptimizer> GlobalOptimizer::create(string type){
    if(type.empty() || type=="g2o")
        return std::make_shared<GlobalOptimizerG2O>();


    else throw std::runtime_error("GlobalOptimizer::create could not load the required optimizer");
}


void GlobalOptimizer::saveToStream(std::ostream &str){

}

void GlobalOptimizer::readFromStream(std::istream &str){

}
}
