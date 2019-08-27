#Project
cmake_minimum_required(VERSION 2.4)
project(spsstereo)

set(CMAKE_BUILD_TYPE Release)
SET(OpenCV_DIR /home/wangyuanqi/opencv-3.4.1/new_build/)
find_package(OpenCV REQUIRED)
find_package( CUDA 10 REQUIRED )
set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")

set(SPSSTEREO_SRC_FILES "src/cpu/*.cpp")
set(SPSSTEREO_CUDA_SRC_FILES "src/gpu/*.cu")
set(SPSSTEREO_INCLUDE_FILES "include/cpu/*.h")
set(SPSSTEREO_CUDA_INCLUDE_FILES "include/gpu/*.h")
#file(GLOB SPSSTEREO_SRC_FILES "src/cpu/*.cpp")
#file(GLOB SPSSTEREO_CUDA_SRC_FILES "src/gpu/*.cu")
#file(GLOB SPSSTEREO_INCLUDE_FILES "include/cpu/*.h")
#file(GLOB SPSSTEREO_CUDA_INCLUDE_FILES "include/gpu/*.h")
#cuda_add_executable(
#        spsstereo
        #include/gpu/cost_aggregation.h include/gpu/new_cost_aggregation.h include/gpu/util.h
        #${SPSSTEREO_CUDA_INCLUDE_FILES}
#        ${SPSSTEREO_CUDA_SRC_FILES}
#        ${SPSSTEREO_SRC_FILES}
#)


INCLUDE_DIRECTORIES("include/")
INCLUDE_DIRECTORIES("include/cpu/")
INCLUDE_DIRECTORIES("include/gpu/")
cuda_add_executable(
        spsstereo
        src/gpu/costs.cu src/gpu/debug.cu src/gpu/depth_compute.cu
        src/gpu/hamming_cost.cu src/gpu/image_warp.cu src/gpu/median_filter.cu
        src/gpu/n_hamming_cost.cu
        src/cpu/SPSStereo.cu src/cpu/spsstereo_main.cu
)

add_executable(spsstereo include/gpu/new_cost_aggregation.h include/gpu/costs.h
        include/gpu/debug.h include/gpu/depth_compute.h include/gpu/hamming_cost.h
        include/gpu/image_warp.h include/gpu/median_filter.h include/gpu/n_hamming_cost.h
        include/gpu/util.h)
# SPS-Stereo

#cuda_add_executable(
#        spsstereo
#        ${SPSSTEREO_CUDA_INCLUDE_FILES}
#        ${SPSSTEREO_SRC_FILES})
#add_executable(spsstereo include/util.h)

#add_executable(spsstereo ${SPSSTEREO_INCLUDE_FILES})
#target_include_directories(spsstereo PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(spsstereo ${OpenCV_LIBRARIES})
