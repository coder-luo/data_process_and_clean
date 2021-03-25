cmake_minimum_required(VERSION 3.15)
project(main)

set(CMAKE_CXX_STANDARD 11)

# 指定libTorch位置
include_directories(${CMAKE_SOURCE_DIR}/3rdParty/libtorch/include)
link_directories(${CMAKE_SOURCE_DIR}/3rdParty/libtorch/lib)

# 指定opencv位置
include_directories(${CMAKE_SOURCE_DIR}/3rdParty/opencv/include)
link_directories(${CMAKE_SOURCE_DIR}/3rdParty/opencv/lib)

# 指定cuda位置
include_directories(/usr/local/cuda-10.1/targets/x86_64-linux/include)
link_directories(/usr/local/cuda-10.1/targets/x86_64-linux/lib)
link_directories(/usr/local/cuda-10.1/targets/x86_64-linux/lib/stubs)


list(APPEND LIBS PUBLIC c10)
list(APPEND LIBS PUBLIC torch)
list(APPEND LIBS PUBLIC nvrtc)
list(APPEND LIBS PUBLIC cuda)
list(APPEND LIBS PUBLIC caffe2)
list(APPEND LIBS PUBLIC caffe2_gpu)

list(APPEND LIBS PUBLIC opencv_core)
list(APPEND LIBS PUBLIC opencv_imgproc)
list(APPEND LIBS PUBLIC opencv_imgcodecs)
list(APPEND LIBS PUBLIC opencv_highgui)
list(APPEND LIBS PUBLIC opencv_ml)
list(APPEND LIBS PUBLIC opencv_video)
list(APPEND LIBS PUBLIC opencv_videoio)
list(APPEND LIBS PUBLIC opencv_calib3d)

include_directories(${CMAKE_SOURCE_DIR}/base)
aux_source_directory(${CMAKE_SOURCE_DIR}/base dirSrc)

add_executable(main main.cpp ${dirSrc})
target_link_libraries(main ${LIBS})
