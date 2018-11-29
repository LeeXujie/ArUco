INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src )
LINK_LIBRARIES(${PROJECT_NAME})

IF(${OPENNI2LIB_FOUND})
add_definitions(-DOPENNI2)
SET(REQUIRED_LIBRARIES ${REQUIRED_LIBRARIES} ${LIBOPENNI2_LIBRARY})
ENDIF()

add_subdirectory(rgbdreader)
include_directories(rgbdreader)

add_library(global_icp global_icp.cpp)

add_executable(rgbdslam rgbdslam.cpp)
target_link_libraries(rgbdslam rgbdreader)

add_library(sequence_optimizer sequenceoptimizer.cpp)

find_package(depthmaps REQUIRED)
#find_package(PCL REQUIRED COMPONENTS common registration)
add_executable(deformable_mapping deformable_mapping.cpp)
target_link_libraries(deformable_mapping ${REQUIRED_LIBRARIES} rgbdreader sequence_optimizer global_icp ${depthmaps_LIBS})

add_executable(read_rgbd read_rgbd.cpp)
target_link_libraries(read_rgbd ${REQUIRED_LIBRARIES} rgbdreader)
